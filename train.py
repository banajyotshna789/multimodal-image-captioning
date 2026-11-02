import os
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from dataset import CaptionDataset, build_vocab_from_csv, caption_collate_fn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Encoder (ResNet50)
# -----------------------------
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.relu = nn.ReLU()

    def forward(self, images):
        features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.relu(self.linear(features))
        return features


# -----------------------------
# Decoder (LSTM)
# -----------------------------
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])
        features = features.unsqueeze(1)
        inputs = torch.cat((features, embeddings), dim=1)
        hiddens, _ = self.lstm(inputs)
        outputs = self.linear(hiddens)
        return outputs


# -----------------------------
# Captioning Model
# -----------------------------
class CaptionModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super().__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs


# -----------------------------
# Training
# -----------------------------
def train():
    csv_file = "train.csv"

    vocab = build_vocab_from_csv(csv_file)
    dataset = CaptionDataset(csv_file, vocab=vocab)
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=caption_collate_fn,
    )

    model = CaptionModel(embed_size=256, hidden_size=256, vocab_size=len(vocab)).to(DEVICE)

    if os.path.exists("caption_model.pth"):
        model.load_state_dict(torch.load("caption_model.pth", map_location=DEVICE))
        print("Loaded saved model checkpoint.")

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10

    for epoch in range(num_epochs):
        total_loss = 0

        for images, captions, _ in loader:
            images = images.to(DEVICE)
            captions = captions.to(DEVICE)

            outputs = model(images, captions)

            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]),
                captions.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "caption_model.pth")
    print("Model saved!")

if __name__ == "__main__":
    train()
