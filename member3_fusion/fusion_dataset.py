import os
import pickle
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


def load_vocab(vocab_path: str):
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    return vocab


def tokenize(text: str):
    return str(text).lower().strip().split()


def resolve_image_path(image_value: str, image_root: Optional[str] = None) -> str:
    image_value = str(image_value)

    if os.path.isabs(image_value) and os.path.exists(image_value):
        return image_value

    if image_root is not None:
        candidate = os.path.join(image_root, image_value)
        if os.path.exists(candidate):
            return candidate

    if os.path.exists(image_value):
        return image_value

    raise FileNotFoundError(f"Image not found: {image_value}")


def load_image(image_path: str, image_root: Optional[str] = None):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    full_path = resolve_image_path(image_path, image_root=image_root)
    image = Image.open(full_path).convert("RGB")
    return transform(image)


def process_caption(caption, vocab, max_len: int = 20) -> torch.Tensor:
    indices = vocab.numericalize(caption)

    pad_idx = vocab.stoi.get("<PAD>", vocab.stoi.get("<pad>", 0))

    if len(indices) < max_len:
        indices += [pad_idx] * (max_len - len(indices))
    else:
        indices = indices[:max_len]

    return torch.tensor(indices, dtype=torch.long)


class FusionDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        vocab_path: str = "vocab.pkl",
        image_root: str = "..",
        max_len: int = 20,
    ):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.vocab = load_vocab(vocab_path)
        self.image_root = image_root
        self.max_len = max_len

        required = {"image_path", "caption", "label"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"{csv_file} is missing columns: {sorted(missing)}")

    @property
    def vocab_size(self):
        return len(self.vocab.stoi)

    @property
    def pad_idx(self):
        return self.vocab.stoi.get("<PAD>", self.vocab.stoi.get("<pad>", 0))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        image_path = str(row["image_path"])
        caption = str(row["caption"])
        label = float(row["label"])

        image = load_image(image_path, self.image_root)
        caption_tensor = process_caption(caption, self.vocab, self.max_len)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        length = int((caption_tensor != self.pad_idx).sum().item())

        return image, caption_tensor, length, label_tensor
