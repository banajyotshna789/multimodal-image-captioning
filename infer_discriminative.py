import torch

from discriminative_dataset import load_vocab, load_image, process_caption
from discriminative_model import DiscriminativeModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load vocab
vocab = load_vocab("vocab.pkl")

# Load model
model = DiscriminativeModel(
    vocab_size=len(vocab.stoi),
    pad_idx=vocab.pad_idx,
).to(device)

checkpoint_path = "discriminative_model_epoch10.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()


def get_score(image_path, caption, image_root="data/images/small"):
    image_tensor = load_image(image_path, image_root=image_root).unsqueeze(0).to(device)
    caption_tensor = process_caption(caption, vocab=vocab, max_len=20).unsqueeze(0).to(device)
    length_tensor = torch.tensor(
        [min(len(vocab.numericalize(caption)), 20)],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        logit = model(image_tensor, caption_tensor, length_tensor)
        score = torch.sigmoid(logit).item()

    return score


if __name__ == "__main__":
    # Example image
    image_path = "cb/cb728deb.jpg"

    # Reference caption for evaluation only
    true_caption = "women black shoes"

    # Generated captions from your generative models
    lstm_caption = "men brown leather boots"
    transformer_caption = "women black leather boots"

    print(f"Image: {image_path}\n")

    # Print true caption score for analysis only
    true_score = get_score(image_path, true_caption)
    print(f"True caption: {true_caption}")
    print(f"Score: {true_score:.4f}\n")

    # Only rerank generated captions for final decision
    generated_captions = [
        ("LSTM caption", lstm_caption),
        ("Transformer caption", transformer_caption),
    ]

    scores = []
    for name, cap in generated_captions:
        score = get_score(image_path, cap)
        scores.append((name, cap, score))
        print(f"{name}: {cap}")
        print(f"Score: {score:.4f}\n")

    best = max(scores, key=lambda x: x[2])
    print(f"Best generated caption: {best[1]}")
    print(f"Best source: {best[0]}")
    print(f"Best score: {best[2]:.4f}")
