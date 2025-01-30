import os
import sys

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from ...project.logger import logger
from ..config import clip_config


def load_clip_model(pretrained_model_name="openai/clip-vit-base-patch32", device="cpu"):
    model = CLIPModel.from_pretrained(pretrained_model_name).to(device)
    processor = CLIPProcessor.from_pretrained(pretrained_model_name)
    return model, processor


def get_image_embeddings(
    images,
    desired_image_number=clip_config.DESIRED_IMAGE_NUMBER,
    device="cpu",
    is_thingsvision=False,  # noqa: FBT002
):
    try:
        # load embeddings from a file, not from scratch
        if clip_config.LOAD_EMBEDDINGS_FILE != "":
            logger.info("Loading embeddings from local file: %s", clip_config.LOAD_EMBEDDINGS_FILE)
            # Load embeddings from configured file instead of recalculating
            if is_thingsvision:  # process thingsvision embeddings with numpy
                logger.info("Loading THINGSvision")
                embeddings = np.load(clip_config.LOAD_EMBEDDINGS_FILE, allow_pickle=True).astype(np.float32)
            else:
                embeddings = np.load(clip_config.LOAD_EMBEDDINGS_FILE)
            embeddings = torch.from_numpy(embeddings).to(device)[
                :desired_image_number
            ]  # Move to device (CPU/GPU) if needed

        else:
            # load from scratch with CLIP model and provided directory
            logger.info("Loading embeddings from scratch")
            logger.info("Loading CLIP Model: %s", clip_config.PRETRAINED_MODEL)

            model, processor = load_clip_model(pretrained_model_name=clip_config.PRETRAINED_MODEL, device=device)
            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
                embeddings = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        # if our embeddings don't process correctly, raise an error and exit
        if embeddings is None or not isinstance(embeddings, torch.Tensor):
            raise ValueError("Failed to generate embeddings.")

    # catch if load embeddings file isn't correct, ValueError as specified above
    except (FileNotFoundError, ValueError) as e:
        logger.error("Error loading CLIP embeddings: %s", e)
        sys.exit(1)

    # log the shape and return the scratch-made embeddings
    logger.info("Embeddings shape: %s", embeddings.shape)
    return embeddings


# Load images from directory
def load_images(root_dir, desired_image_number=clip_config.DESIRED_IMAGE_NUMBER):
    try:
        image_dir = os.path.join(
            f"{root_dir}/{clip_config.ROOT_DATA_DIR}",
            f"subj0{clip_config.SUBJECT}",
            "training_split",
            "training_images",
        )
        if not os.path.exists(image_dir):  # noqa: PTH110
            raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
        image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if os.path.isfile(os.path.join(image_dir, f))  # noqa: PTH113
        ]
        image_paths.sort()
        image_paths = image_paths[:desired_image_number]
        images = [Image.open(path).convert("RGB") for path in image_paths]
        if images is None or len(images) == 0:
            raise ValueError("No images were loaded.")
    except (FileNotFoundError, ValueError) as e:
        logger.error("Error loading CLIP embeddings: %s", e)
        sys.exit(1)
    logger.info("Loaded %s number of images", len(images))
    return images
