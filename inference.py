import pickle
import torch
from PIL import Image
from torchvision import transforms

from train import CaptionModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load vocab
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

# load model
model = CaptionModel(embed_size=256, hidden_size=256, vocab_size=len(vocab)).to(DEVICE)
model.load_state_dict(torch.load("caption_model_v1.pth", map_location=DEVICE))
model.eval()

# image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def generate_caption(model, image, vocab, max_len=20):
    with torch.no_grad():
        feature = model.encoder(image.to(DEVICE))
        sampled_ids = [vocab.stoi["<SOS>"]]
        hidden = None
        inputs = feature.unsqueeze(1)

        for _ in range(max_len):
            if len(sampled_ids) > 1:
                last = torch.tensor([[sampled_ids[-1]]], device=DEVICE)
                inputs = model.decoder.embed(last)

            hiddens, hidden = model.decoder.lstm(inputs, hidden)
            outputs = model.decoder.linear(hiddens[:, -1, :])
            predicted = outputs.argmax(1).item()

            if predicted == vocab.stoi["<EOS>"]:
                break

            sampled_ids.append(predicted)

        words = []
        for idx in sampled_ids[1:]:
            word = vocab.itos.get(idx, "<UNK>")
            if word not in ("<SOS>", "<EOS>", "<PAD>"):
                words.append(word)

        return " ".join(words)

# test image
img = Image.open("data/images/small/cb/cb728deb.jpg").convert("RGB")
img = transform(img).unsqueeze(0)

print(generate_caption(model, img, vocab))
