import os
import pickle
from typing import Optional

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class VocabularyAdapter:
    """
    Wraps your existing dataset.Vocabulary object and exposes a few
    convenience properties used by the discriminative pipeline.
    """

    def __init__(self, vocab_obj):
        self.vocab = vocab_obj

    def __getattr__(self, name):
        return getattr(self.vocab, name)

    @property
    def pad_idx(self) -> int:
        if hasattr(self.vocab, "stoi"):
            return self.vocab.stoi.get("<PAD>", self.vocab.stoi.get("<pad>", 0))
        return 0

    @property
    def unk_idx(self) -> int:
        if hasattr(self.vocab, "stoi"):
            return self.vocab.stoi.get("<UNK>", self.vocab.stoi.get("<unk>", 3))
        return 3

    @property
    def bos_idx(self) -> int:
        if hasattr(self.vocab, "stoi"):
            return self.vocab.stoi.get("<BOS>", self.vocab.stoi.get("<bos>", self.pad_idx))
        return self.pad_idx

    @property
    def eos_idx(self) -> int:
        if hasattr(self.vocab, "stoi"):
            return self.vocab.stoi.get("<EOS>", self.vocab.stoi.get("<eos>", self.pad_idx))
        return self.pad_idx


def load_vocab(vocab_path: str = "vocab.pkl") -> VocabularyAdapter:
    with open(vocab_path, "rb") as f:
        vocab_obj = pickle.load(f)
    return VocabularyAdapter(vocab_obj)


def tokenize(text: str):
    return text.lower().strip().split()


def resolve_image_path(image_value: str, image_root: Optional[str] = None) -> str:
    if os.path.isabs(image_value) and os.path.exists(image_value):
        return image_value

    if image_root:
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
    """
    Converts caption text to a padded tensor of token indices.
    Uses the same vocab.numericalize() method as your main project.
    """
    indices = vocab.numericalize(caption)

    if len(indices) < max_len:
        indices += [vocab.pad_idx] * (max_len - len(indices))
    else:
        indices = indices[:max_len]

    return torch.tensor(indices, dtype=torch.long)


class DiscriminativeDataset(Dataset):
    """
    Expected CSV columns:
        image_path, caption, label

    For compatibility, this also accepts:
        image, caption, label
    """

    def __init__(
        self,
        csv_file: str,
        vocab_path: str = "vocab.pkl",
        image_root: Optional[str] = "data/images/small",
        max_len: int = 20,
    ):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.vocab = load_vocab(vocab_path)
        self.image_root = image_root
        self.max_len = max_len

        cols = set(self.df.columns)
        if "image_path" not in cols and "image" not in cols:
            raise ValueError(
                f"{csv_file} is missing an image column. Expected 'image_path' or 'image'. "
                f"Found: {sorted(cols)}"
            )

        required_cols = {"caption", "label"}
        missing = required_cols - cols
        if missing:
            raise ValueError(f"{csv_file} is missing columns: {sorted(missing)}")

    @property
    def vocab_size(self) -> int:
        if hasattr(self.vocab, "stoi"):
            return len(self.vocab.stoi)
        return len(self.vocab)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        if "image_path" in row:
            image_path = str(row["image_path"])
        else:
            image_path = str(row["image"])

        caption = str(row["caption"])
        label = float(row["label"])

        image = load_image(image_path, self.image_root)
        caption_tensor = process_caption(caption, vocab=self.vocab, max_len=self.max_len)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return image, caption_tensor, label_tensor


def collate_batch(batch):
    """
    Returns:
        images: FloatTensor [B, C, H, W]
        captions: LongTensor [B, T]
        lengths: LongTensor [B]
        labels: FloatTensor [B]
    """
    images, captions, labels = zip(*batch)

    images = torch.stack(images, dim=0)
    labels = torch.stack(labels, dim=0)

    pad_idx = 0
    lengths = torch.tensor(
        [(cap != pad_idx).sum().item() for cap in captions],
        dtype=torch.long
    )

    captions = pad_sequence(captions, batch_first=True, padding_value=pad_idx)

    return images, captions, lengths, labels
