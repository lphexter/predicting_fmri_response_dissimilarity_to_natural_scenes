# clip_utils.py

import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def load_clip_model(pretrained_model_name="openai/clip-vit-base-patch32", device="cpu"):
    """
    Loads a CLIP model + processor from Hugging Face.
    """
    model = CLIPModel.from_pretrained(pretrained_model_name).to(device)
    processor = CLIPProcessor.from_pretrained(pretrained_model_name)
    return model, processor

def get_image_embeddings(image_dir, processor, model, desired_image_number=500, device="cpu"):
    """
    Extract normalized image embeddings from CLIP.
    """
    # Collect and sort image paths
    image_paths = [os.path.join(image_dir, f)
                   for f in os.listdir(image_dir)
                   if os.path.isfile(os.path.join(image_dir, f))]
    image_paths.sort()
    image_paths = image_paths[:desired_image_number]

    # Load images
    images = [Image.open(path).convert("RGB") for path in image_paths]

    # Preprocess
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        # Extract image features
        image_features = model.get_image_features(**inputs)
        # Normalize
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

    return image_features
