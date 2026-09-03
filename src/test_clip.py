import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("Forensic Media Search - Attribute Test")
print("=" * 70)

print(f"GPU: {torch.cuda.get_device_name(0)}")

model, preprocess = clip.load(
    "ViT-B/32",
    device=device
)

model.eval()

image_path = "/evidence/casa.jpg"

image = preprocess(
    Image.open(image_path).convert("RGB")
).unsqueeze(0).to(device)


def classify(queries, title):

    text = clip.tokenize(queries).to(device)

    with torch.no_grad():

        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        image_features /= image_features.norm(
            dim=-1,
            keepdim=True
        )

        text_features /= text_features.norm(
            dim=-1,
            keepdim=True
        )

        # CLIP posee un factor de escala aprendido
        logit_scale = model.logit_scale.exp()

        logits = (
            logit_scale *
            image_features @ text_features.T
        )

        probabilities = logits.softmax(dim=-1)[0]

        cosine = (
            image_features @ text_features.T
        )[0]

    results = []

    for query, score, probability in zip(
        queries,
        cosine.cpu().numpy(),
        probabilities.cpu().numpy()
    ):
        results.append(
            (query, score, probability)
        )

    results.sort(
        key=lambda x: x[2],
        reverse=True
    )

    print()
    print(title)
    print("-" * 70)

    for query, score, probability in results:

        print(
            f"{probability * 100:6.2f}% "
            f"cos={score:.4f} "
            f"{query}"
        )


classify(
    [
        "a house with a red roof",
        "a house with a black roof",
        "a house with a gray roof",
        "a house with a white roof",
        "a house with a brown roof",
        "a house with a green roof",
        "a house with a blue roof",
    ],
    "ROOF COLOR"
)


classify(
    [
        "a house with a black door",
        "a house with a white door",
        "a house with a red door",
        "a house with a brown door",
        "a house with a gray door",
        "a house with a blue door",
    ],
    "DOOR COLOR"
)