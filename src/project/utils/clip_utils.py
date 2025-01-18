import os

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def load_clip_model(pretrained_model_name="openai/clip-vit-base-patch32", device="cpu"):
    model = CLIPModel.from_pretrained(pretrained_model_name).to(device)
    processor = CLIPProcessor.from_pretrained(pretrained_model_name)
    return model, processor


def get_image_embeddings(image_dir, processor, model, desired_image_number=500, device="cpu"):
    image_paths = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, f))  # noqa: PTH113
    ]
    image_paths.sort()
    image_paths = image_paths[:desired_image_number]

    images = [Image.open(path).convert("RGB") for path in image_paths]
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        return image_features / image_features.norm(p=2, dim=-1, keepdim=True)
