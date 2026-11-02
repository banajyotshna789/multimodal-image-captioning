import torch
import pandas as pd
from PIL import Image
from torchvision import transforms

from dataset import build_vocab_from_csv
from train import CaptionModel as LSTMModel
from train_transformer import CaptionModel as TransformerModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_lstm(model, image, vocab):
    model.eval()
    with torch.no_grad():
        feature = model.encoder(image.to(DEVICE))

        sampled_ids = [vocab.stoi["<SOS>"]]
        hidden = None
        inputs = feature.unsqueeze(1)

        for _ in range(20):
            if len(sampled_ids) > 1:
                last_word = torch.tensor([[sampled_ids[-1]]], device=DEVICE)
                inputs = model.decoder.embed(last_word)

            hiddens, hidden = model.decoder.lstm(inputs, hidden)
            outputs = model.decoder.linear(hiddens[:, -1, :])
            predicted = outputs.argmax(1).item()

            if predicted == vocab.stoi["<EOS>"]:
                break

            sampled_ids.append(predicted)

        words = []
        for idx in sampled_ids[1:]:
            word = vocab.itos.get(idx, "<UNK>")
            if word in ("<SOS>", "<EOS>", "<PAD>"):
                continue
            words.append(word)

        return " ".join(words)


def generate_transformer(model, image, vocab):
    model.eval()
    with torch.no_grad():
        feature = model.encoder(image.to(DEVICE))

        sampled_ids = [vocab.stoi["<SOS>"]]

        for _ in range(20):
            captions = torch.tensor(sampled_ids, device=DEVICE).unsqueeze(0)

            if captions.size(1) == 1:
                captions = torch.cat(
                    [captions, torch.tensor([[vocab.stoi["<PAD>"]]], device=DEVICE)],
                    dim=1
                )

            outputs = model.decoder(feature, captions)
            predicted = outputs[0, -1, :].argmax().item()

            if predicted == vocab.stoi["<EOS>"]:
                break

            sampled_ids.append(predicted)

        words = []
        for idx in sampled_ids[1:]:
            word = vocab.itos.get(idx, "<UNK>")
            if word in ("<SOS>", "<EOS>", "<PAD>"):
                continue
            words.append(word)

        # remove duplicates
        clean = []
        for w in words:
            if not clean or clean[-1] != w:
                clean.append(w)

        return " ".join(clean)


def main():
    vocab = build_vocab_from_csv("train.csv")

    lstm = LSTMModel(embed_size=256, hidden_size=256, vocab_size=len(vocab)).to(DEVICE)
    lstm.load_state_dict(torch.load("caption_model.pth", map_location=DEVICE))

    transformer = TransformerModel(embed_size=256, vocab_size=len(vocab)).to(DEVICE)
    transformer.load_state_dict(torch.load("transformer_model.pth", map_location=DEVICE))

    df = pd.read_csv("test.csv")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    for i in range(5):
        row = df.iloc[i]
        image = Image.open(row["image_path"]).convert("RGB")
        image = transform(image).unsqueeze(0)

        lstm_pred = generate_lstm(lstm, image, vocab)
        trans_pred = generate_transformer(transformer, image, vocab)

        print("\n============================")
        print("Image:", row["image_path"])
        print("True :", row["caption"])
        print("LSTM :", lstm_pred)
        print("Trans:", trans_pred)


if __name__ == "__main__":
    main()
