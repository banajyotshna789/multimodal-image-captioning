import argparse
import copy
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torchvision import models

from fusion_dataset import FusionDataset
from fusion_models import NoFusionBaseline, ConcatFusion, GatedFusion


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ImageEncoder(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        try:
            backbone = models.resnet50(pretrained=pretrained)
        except TypeError:
            backbone = models.resnet50(weights=None)
        except Exception:
            backbone = models.resnet50(pretrained=False)

        self.features = nn.Sequential(*list(backbone.children())[:-1])  # [B, 2048, 1, 1]

    def forward(self, images):
        x = self.features(images)
        x = x.squeeze(-1).squeeze(-1)  # [B, 2048]
        return x


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, pad_idx=0, embed_dim=256, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, captions, lengths):
        x = self.embedding(captions)  # [B, T, E]
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        txt_feat = torch.cat((hidden[-2], hidden[-1]), dim=1)  # [B, 512]
        return txt_feat


def build_model(model_name: str):
    model_name = model_name.lower()
    if model_name == "baseline":
        return NoFusionBaseline(img_dim=2048, txt_dim=512, hidden_dim=256)
    elif model_name == "concat":
        return ConcatFusion(img_dim=2048, txt_dim=512, hidden_dim=512)
    elif model_name == "gated":
        return GatedFusion(img_dim=2048, txt_dim=512, fused_dim=512)
    else:
        raise ValueError("Unknown model name: {}".format(model_name))


def compute_metrics(y_true, y_prob):
    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    return acc, prec, rec, f1, auc


@torch.no_grad()
def evaluate(model, image_encoder, text_encoder, loader, device):
    model.eval()
    image_encoder.eval()
    text_encoder.eval()

    probs_all = []
    labels_all = []

    for images, captions, lengths, labels in loader:
        images = images.to(device)
        captions = captions.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        img_feat = image_encoder(images)        # [B, 2048]
        txt_feat = text_encoder(captions, lengths)  # [B, 512]

        logits = model(img_feat, txt_feat)
        probs = torch.sigmoid(logits)

        probs_all.extend(probs.cpu().tolist())
        labels_all.extend(labels.cpu().tolist())

    return compute_metrics(labels_all, probs_all)


def train_one_epoch(model, image_encoder, text_encoder, loader, optimizer, criterion, device):
    model.train()
    image_encoder.eval()
    text_encoder.eval()

    total_loss = 0.0

    for images, captions, lengths, labels in loader:
        images = images.to(device)
        captions = captions.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            img_feat = image_encoder(images)
            txt_feat = text_encoder(captions, lengths)

        logits = model(img_feat, txt_feat)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gated", choices=["baseline", "concat", "gated"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", type=str, default="discriminative_data.csv")
    parser.add_argument("--vocab", type=str, default="vocab.pkl")
    parser.add_argument("--image_root", type=str, default="..")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.save_dir, exist_ok=True)

    dataset = FusionDataset(
        csv_file=args.csv,
        vocab_path=args.vocab,
        image_root=args.image_root,
        max_len=20,
    )

    # 80/10/10 split
    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val

    train_set, val_set, test_set = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    image_encoder = ImageEncoder(pretrained=True).to(device)
    text_encoder = TextEncoder(
        vocab_size=dataset.vocab_size,
        pad_idx=dataset.pad_idx,
        embed_dim=256,
        hidden_dim=256,
    ).to(device)

    # Freeze encoders: train only fusion classifier
    for p in image_encoder.parameters():
        p.requires_grad = False
    for p in text_encoder.parameters():
        p.requires_grad = False

    model = build_model(args.model).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_auc = -1.0
    best_state = None

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model=model,
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_acc, val_prec, val_rec, val_f1, val_auc = evaluate(
            model, image_encoder, text_encoder, val_loader, device
        )

        print("Epoch {}/{}".format(epoch + 1, args.epochs))
        print("Train Loss: {:.4f}".format(train_loss))
        print("Val   Acc : {:.4f}".format(val_acc))
        print("Val   Prec: {:.4f}".format(val_prec))
        print("Val   Rec : {:.4f}".format(val_rec))
        print("Val   F1  : {:.4f}".format(val_f1))
        print("Val   AUC : {:.4f}".format(val_auc))
        print("-" * 60)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc, test_prec, test_rec, test_f1, test_auc = evaluate(
        model, image_encoder, text_encoder, test_loader, device
    )

    print("\n=== FINAL TEST RESULTS ({}) ===".format(args.model.upper()))
    print("Test Acc : {:.4f}".format(test_acc))
    print("Test Prec: {:.4f}".format(test_prec))
    print("Test Rec : {:.4f}".format(test_rec))
    print("Test F1  : {:.4f}".format(test_f1))
    print("Test AUC : {:.4f}".format(test_auc))

    ckpt_path = os.path.join(args.save_dir, "{}.pth".format(args.model))
    torch.save(model.state_dict(), ckpt_path)
    print("Saved checkpoint to {}".format(ckpt_path))


if __name__ == "__main__":
    main()
