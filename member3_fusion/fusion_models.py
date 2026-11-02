import torch
import torch.nn as nn


class NoFusionBaseline(nn.Module):
    """
    Baseline model:
    - image branch and text branch are scored separately
    - final score is the average of both logits
    """
    def __init__(self, img_dim=2048, txt_dim=512, hidden_dim=256):
        super().__init__()

        self.img_head = nn.Sequential(
            nn.Linear(img_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

        self.txt_head = nn.Sequential(
            nn.Linear(txt_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, img_feat, txt_feat):
        img_logit = self.img_head(img_feat)
        txt_logit = self.txt_head(txt_feat)
        return ((img_logit + txt_logit) / 2.0).squeeze(1)


class ConcatFusion(nn.Module):
    """
    Simple concatenation-based fusion.
    """
    def __init__(self, img_dim=2048, txt_dim=512, hidden_dim=512):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(img_dim + txt_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, img_feat, txt_feat):
        x = torch.cat([img_feat, txt_feat], dim=1)
        return self.classifier(x).squeeze(1)


class GatedFusion(nn.Module):
    """
    Gated fusion:
    learns how much to trust image vs text features.
    """
    def __init__(self, img_dim=2048, txt_dim=512, fused_dim=512):
        super().__init__()

        self.img_proj = nn.Linear(img_dim, fused_dim)
        self.txt_proj = nn.Linear(txt_dim, fused_dim)

        self.gate = nn.Sequential(
            nn.Linear(fused_dim * 2, fused_dim),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fused_dim // 2, 1),
        )

    def forward(self, img_feat, txt_feat):
        img_proj = self.img_proj(img_feat)
        txt_proj = self.txt_proj(txt_feat)

        gate = self.gate(torch.cat([img_proj, txt_proj], dim=1))
        fused = gate * img_proj + (1.0 - gate) * txt_proj

        return self.classifier(fused).squeeze(1)
