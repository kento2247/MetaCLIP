import torch
from PIL import Image
from src.mini_clip.factory import create_model_and_transforms, get_tokenizer


model, _, preprocess = create_model_and_transforms(
    "ViT-H-14-quickgelu-worldwide@WorldWideCLIP", pretrained="metaclip2_worldwide"
)
tokenize = get_tokenizer("facebook/xlm-v-base")

image = preprocess(Image.open("docs/CLIP.png")).unsqueeze(0)
text = tokenize(["a diagram", "a dog", "a cat"])

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)
