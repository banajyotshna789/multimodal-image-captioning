import torch
import torch.nn as nn
import torchvision.models as models


class DiscriminativeModel(nn.Module):
    def __init__(self, vocab_size, pad_idx=0, emb_dim=256, hidden_dim=256):
        super().__init__()

        # Image encoder
        backbone = models.resnet18(pretrained=False)
        self.image_encoder = nn.Sequential(*list(backbone.children())[:-1])  # [B, 512, 1, 1]
        self.image_proj = nn.Linear(512, hidden_dim)

        # Text encoder
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.text_encoder = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.text_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, images, captions, lengths):
        # Image branch
        img_feat = self.image_encoder(images).squeeze(-1).squeeze(-1)  # [B, 512]
        img_feat = self.image_proj(img_feat)  # [B, H]

        # Text branch
        cap_emb = self.embedding(captions)  # [B, T, E]
        packed = nn.utils.rnn.pack_padded_sequence(
            cap_emb,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.text_encoder(packed)

        # Bidirectional hidden states
        cap_feat = torch.cat((hidden[-2], hidden[-1]), dim=1)  # [B, 2H]
        cap_feat = self.text_proj(cap_feat)  # [B, H]

        # Fusion
        fused = torch.cat((img_feat, cap_feat), dim=1)  # [B, 2H]
        logits = self.classifier(fused).squeeze(1)  # [B]
        return logits

    @torch.no_grad()
    def score_candidates(self, image_tensor, caption_tensors, length_tensors):
        """
        Scores multiple captions for the same image.

        Args:
            image_tensor: Tensor [1, C, H, W]
            caption_tensors: list of Tensor [1, T]
            length_tensors: list of int or Tensor
        Returns:
            list of float scores in [0, 1]
        """
        self.eval()
        scores = []

        for caption_tensor, length in zip(caption_tensors, length_tensors):
            if isinstance(length, int):
                length = torch.tensor([length], device=image_tensor.device)
            elif isinstance(length, torch.Tensor) and length.dim() == 0:
                length = length.unsqueeze(0)

            logits = self.forward(image_tensor, caption_tensor, length)
            score = torch.sigmoid(logits).item()
            scores.append(score)

        return scores
