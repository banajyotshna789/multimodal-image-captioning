import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import os

from dataset import CaptionDataset, build_vocab_from_csv, caption_collate_fn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Encoder (same as before)
# ---------------------------
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.linear(features)
        return features


# ---------------------------
# Transformer Decoder
# ---------------------------
class TransformerDecoder(nn.Module):
    def __init__(self, embed_size, vocab_size, num_heads=4, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos = nn.Parameter(torch.randn(1, 100, embed_size))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=num_heads
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, features, captions):
        # captions: [batch, seq]
        x = self.embed(captions[:, :-1])  # [batch, seq-1, embed]
        x = x + self.pos[:, :x.size(1), :]

        # Transformer expects [seq, batch, embed]
        x = x.permute(1, 0, 2)

        # causal mask to prevent looking ahead
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(x.size(0)).to(x.device)

        memory = features.unsqueeze(0)  # [1, batch, embed]
        out = self.transformer(x, memory, tgt_mask=tgt_mask)

        out = out.permute(1, 0, 2)  # [batch, seq, embed]
        out = self.fc(out)
        return out

# ---------------------------
# Full Model
# ---------------------------
class CaptionModel(nn.Module):
    def __init__(self, embed_size, vocab_size):
        super().__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = TransformerDecoder(embed_size, vocab_size)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs


# ---------------------------
# Training
# ---------------------------
def train():
    csv_file = "train.csv"

    vocab = build_vocab_from_csv(csv_file)
    dataset = CaptionDataset(csv_file, vocab=vocab)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=caption_collate_fn)

    model = CaptionModel(embed_size=256, vocab_size=len(vocab)).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10

    for epoch in range(num_epochs):
        total_loss = 0

        for images, captions, _ in loader:
            images = images.to(DEVICE)
            captions = captions.to(DEVICE)

            outputs = model(images, captions)
            targets = captions[:, 1:]

            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]),
                targets.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Transformer] Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
    torch.save(model.state_dict(), "transformer_model.pth")
    print("Transformer model saved!")


if __name__ == "__main__":
    train()
