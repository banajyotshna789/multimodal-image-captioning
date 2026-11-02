import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_curve, auc
from torchvision import models

from fusion_dataset import FusionDataset
from fusion_models import NoFusionBaseline, ConcatFusion, GatedFusion


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        try:
            backbone = models.resnet50(pretrained=pretrained)
        except TypeError:
            # newer torchvision versions
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            backbone = models.resnet50(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, images):
        x = self.features(images)
        return x.squeeze(-1).squeeze(-1)  # [B, 2048]


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
        x = self.embedding(captions)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        return torch.cat((hidden[-2], hidden[-1]), dim=1)  # [B, 512]


def build_model(name):
    name = name.lower()
    if name == "baseline":
        return NoFusionBaseline(img_dim=2048, txt_dim=512, hidden_dim=256)
    elif name == "concat":
        return ConcatFusion(img_dim=2048, txt_dim=512, hidden_dim=512)
    elif name == "gated":
        return GatedFusion(img_dim=2048, txt_dim=512, fused_dim=512)
    else:
        raise ValueError("Unknown model: {}".format(name))


@torch.no_grad()
def get_probs(model, image_encoder, text_encoder, loader):
    model.eval()
    image_encoder.eval()
    text_encoder.eval()

    probs_all = []
    labels_all = []

    for images, captions, lengths, labels in loader:
        images = images.to(device)
        captions = captions.to(device)
        lengths = lengths.to(device)

        img_feat = image_encoder(images)
        txt_feat = text_encoder(captions, lengths)

        logits = model(img_feat, txt_feat)
        probs = torch.sigmoid(logits)

        probs_all.extend(probs.cpu().numpy().tolist())
        labels_all.extend(labels.numpy().tolist())

    return np.array(labels_all), np.array(probs_all)


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    dataset = FusionDataset(
        csv_file="discriminative_data.csv",
        vocab_path="vocab.pkl",
        image_root="..",
        max_len=20,
    )

    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val

    _, _, test_set = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    # IMPORTANT: use pretrained=True to match training setup
    image_encoder = ImageEncoder(pretrained=True).to(device)
    text_encoder = TextEncoder(
        vocab_size=dataset.vocab_size,
        pad_idx=dataset.pad_idx,
        embed_dim=256,
        hidden_dim=256,
    ).to(device)

    for p in image_encoder.parameters():
        p.requires_grad = False
    for p in text_encoder.parameters():
        p.requires_grad = False

    models_to_plot = {
        "Baseline": ("checkpoints/baseline.pth", build_model("baseline")),
        "Concat": ("checkpoints/concat.pth", build_model("concat")),
        "Gated": ("checkpoints/gated.pth", build_model("gated")),
    }

    plt.figure(figsize=(8, 6))

    for name, (ckpt_path, model) in models_to_plot.items():
        model = model.to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        y_true, y_prob = get_probs(model, image_encoder, text_encoder, test_loader)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label="{} (AUC = {:.4f})".format(name, roc_auc))

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Fusion Models")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("auc_graphs.png", dpi=300)
    print("Saved graph to auc_graphs.png")


if __name__ == "__main__":
    main()
