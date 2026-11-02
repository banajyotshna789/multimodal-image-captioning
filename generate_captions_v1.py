import pickle
import torch
from PIL import Image
from torchvision import transforms

from train import CaptionModel as LSTMCaptionModel
from train_transformer import CaptionModel as TransformerCaptionModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load vocabulary
# -----------------------------
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

# Helpful special token lookup
def get_token_idx(token, fallback=None):
    if hasattr(vocab, "stoi") and token in vocab.stoi:
        return vocab.stoi[token]
    if fallback is not None:
        return fallback
    raise KeyError(f"Token {token} not found in vocab.")

SOS_IDX = get_token_idx("<SOS>", fallback=get_token_idx("<sos>", fallback=0))
EOS_IDX = get_token_idx("<EOS>", fallback=get_token_idx("<eos>", fallback=0))
PAD_IDX = get_token_idx("<PAD>", fallback=get_token_idx("<pad>", fallback=0))

# -----------------------------
# Load models
# -----------------------------
lstm_model = LSTMCaptionModel(embed_size=256, hidden_size=256, vocab_size=len(vocab)).to(DEVICE)
lstm_model.load_state_dict(torch.load("caption_model_v1.pth", map_location=DEVICE))
lstm_model.eval()

transformer_model = TransformerCaptionModel(embed_size=256, vocab_size=len(vocab)).to(DEVICE)
transformer_model.load_state_dict(torch.load("transformer_model_v1.pth", map_location=DEVICE))
transformer_model.eval()

# -----------------------------
# Image transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# -----------------------------
# LSTM caption generation
# -----------------------------
def generate_lstm_caption(model, image, max_len=20):
    with torch.no_grad():
        feature = model.encoder(image.to(DEVICE))
        sampled_ids = [SOS_IDX]
        hidden = None
        inputs = feature.unsqueeze(1)

        for _ in range(max_len):
            if len(sampled_ids) > 1:
                last = torch.tensor([[sampled_ids[-1]]], device=DEVICE)
                inputs = model.decoder.embed(last)

            hiddens, hidden = model.decoder.lstm(inputs, hidden)
            outputs = model.decoder.linear(hiddens[:, -1, :])
            predicted = outputs.argmax(1).item()

            if predicted == EOS_IDX:
                break

            sampled_ids.append(predicted)

        words = []
        for idx in sampled_ids[1:]:
            word = vocab.itos.get(idx, "<UNK>")
            if word not in ("<SOS>", "<EOS>", "<PAD>", "<sos>", "<eos>", "<pad>"):
                words.append(word)

        return " ".join(words)

# -----------------------------
# Transformer caption generation
def generate_transformer_caption(model, image, max_len=20):
    with torch.no_grad():
        image = image.to(DEVICE)
        sampled_ids = [SOS_IDX]

        for _ in range(max_len):
            # make sure input length >= 2
            decoder_input = torch.tensor(
                [sampled_ids + [PAD_IDX]],
                dtype=torch.long,
                device=DEVICE
            )

            outputs = model(image, decoder_input)

            if outputs.size(1) == 0:
                print("Warning: empty output from transformer")
                break

            next_token = outputs[:, -1, :].argmax(dim=1).item()

            if next_token == EOS_IDX:
                break

            sampled_ids.append(next_token)

        words = []
        for idx in sampled_ids[1:]:
            word = vocab.itos.get(idx, "<UNK>")
            if word not in ("<SOS>", "<EOS>", "<PAD>"):
                words.append(word)

        return " ".join(words)# -----------------------------

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    image_paths = [
        "data/images/small/48/48578a4c.jpg",
        "data/images/small/06/06892f5c.jpg",
        "data/images/small/bc/bc3789c2.jpg",
        "data/images/small/22/22161c90.jpg",
        "data/images/small/76/766fd9eb.jpg",
    ]

    for image_path in image_paths:
        image = load_image(image_path)

        lstm_caption = generate_lstm_caption(lstm_model, image)
        transformer_caption = generate_transformer_caption(transformer_model, image)

        print(f"Image: {image_path}")
        print(f"LSTM v1:        {lstm_caption}")
        print(f"Transformer v1:  {transformer_caption}")
        print("-" * 60)
