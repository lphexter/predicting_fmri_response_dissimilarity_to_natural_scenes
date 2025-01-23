import logging
import os

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from ..config.clip_config import LOAD_EMBEDDINGS_FILE, PRETRAINED_MODEL


def load_clip_model(pretrained_model_name="openai/clip-vit-base-patch32", device="cpu"):
    model = CLIPModel.from_pretrained(pretrained_model_name).to(device)
    processor = CLIPProcessor.from_pretrained(pretrained_model_name)
    return model, processor


def get_image_embeddings(image_dir, desired_image_number=500, device="cpu"):
    # initialize logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if LOAD_EMBEDDINGS_FILE != "":
        logging.info("Loading embeddings from local file: %s", LOAD_EMBEDDINGS_FILE)
        # Load embeddings from configued file instead of recalculating
        embeddings = torch.from_numpy(np.load(LOAD_EMBEDDINGS_FILE)).to(device)  # Move to device (CPU/GPU) if needed
        return embeddings[:desired_image_number]
    # load from scratch with CLIP model and provided directory
    logging.info("Loading embeddings from scratch, from image_dir: %s", image_dir)
    logging.info("Loading CLIP Model: %s", PRETRAINED_MODEL)
    model, processor = load_clip_model(pretrained_model_name=PRETRAINED_MODEL, device=device)

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
