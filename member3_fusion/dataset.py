import os
import re
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


def simple_tokenize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.freq_threshold = freq_threshold
        self.itos = {
            0: "<PAD>",
            1: "<SOS>",
            2: "<EOS>",
            3: "<UNK>",
        }
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    def tokenizer(self, text):
        return simple_tokenize(text)

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                frequencies[word] = frequencies.get(word, 0) + 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer(text)
        return [
            self.stoi.get(token, self.stoi["<UNK>"])
            for token in tokenized_text
        ]


class CaptionDataset(Dataset):
    def __init__(self, csv_file, transform=None, vocab=None, max_len=20):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.vocab = vocab
        self.max_len = max_len

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = row["image_path"]
        caption = str(row["caption"])

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        tokens = simple_tokenize(caption)

        if self.vocab is not None:
            caption_num = [self.vocab.stoi["<SOS>"]]
            caption_num += self.vocab.numericalize(caption)[: self.max_len]
            caption_num += [self.vocab.stoi["<EOS>"]]
            caption_num = torch.tensor(caption_num, dtype=torch.long)
            return image, caption_num, caption

        return image, caption


def build_vocab_from_csv(csv_file, freq_threshold=2):
    df = pd.read_csv(csv_file)
    captions = df["caption"].tolist()
    vocab = Vocabulary(freq_threshold=freq_threshold)
    vocab.build_vocabulary(captions)
    return vocab


def caption_collate_fn(batch):
    images = []
    captions = []
    raw_captions = []

    for item in batch:
        if len(item) == 3:
            image, caption_num, raw_caption = item
            images.append(image)
            captions.append(caption_num)
            raw_captions.append(raw_caption)
        else:
            image, raw_caption = item
            images.append(image)
            raw_captions.append(raw_caption)

    images = torch.stack(images, dim=0)

    if len(captions) > 0:
        captions = torch.nn.utils.rnn.pad_sequence(
            captions,
            batch_first=True,
            padding_value=0,
        )
        return images, captions, raw_captions

    return images, raw_captions
