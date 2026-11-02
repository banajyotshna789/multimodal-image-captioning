import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score

from discriminative_dataset import DiscriminativeDataset, collate_batch
from discriminative_model import DiscriminativeModel

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = DiscriminativeDataset(
    csv_file="discriminative_data.csv",
    vocab_path="vocab.pkl",
    image_root="data/images/small",
    max_len=20,
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_batch,
)

model = DiscriminativeModel(
    vocab_size=dataset.vocab_size,
    pad_idx=dataset.vocab.pad_idx,
).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

import os

checkpoint_path = "discriminative_model_v1.pth"

if os.path.exists(checkpoint_path):
    print("Loading previous model (v1)...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

def evaluate(model, loader):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for images, captions, lengths, lab in loader:
            images = images.to(device)
            captions = captions.to(device)
            lengths = lengths.to(device)
            lab = lab.to(device)

            out = model(images, captions, lengths)
            prob = torch.sigmoid(out)

            preds.extend(prob.cpu().numpy().tolist())
            labels.extend(lab.cpu().numpy().tolist())

    preds_bin = [1 if p > 0.5 else 0 for p in preds]

    acc = accuracy_score(labels, preds_bin)
    try:
        auc = roc_auc_score(labels, preds)
    except ValueError:
        auc = 0.0

    return acc, auc


num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for images, captions, lengths, labels in loader:
        images = images.to(device)
        captions = captions.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        outputs = model(images, captions, lengths)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    acc, auc = evaluate(model, loader)
    print(
        f"Epoch {epoch + 1}, Loss: {total_loss:.4f}, "
        f"Acc: {acc:.4f}, AUC: {auc:.4f}"
    )

torch.save(
    model.state_dict(),
    "discriminative_model_epoch10.pth",
    _use_new_zipfile_serialization=False
)
print("Saved discriminative_model_epoch10.pth")
