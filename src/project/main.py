# main.py

"""main Module

This module serves as the entry point for running the CLIP model implementation.

Functions:
    main(): The main function that starts the application.
"""

import argparse
import os
import sys

import torch

from .config import clip_config
from .logger import logger
from .utils.clip_utils import get_image_embeddings, load_images
from .utils.data_utils import (
    analyze_rdm,
    classify_images_rgb,
    create_binary_rdm,
    create_rdm,
    get_equal_color_data,
    load_color_map_files,
    prepare_fmri_data,
    preprocess_images,
)
from .utils.pytorch_training import run_kfold_cv, run_svm
from .utils.visualizations import (
    display_images_by_label,
    plot_rdm_distribution,
    plot_rdm_submatrix,
    print_image_counts,
)


def main():
    """Main function for running the pipeline for predicting fMRI response dissimilarity to pairs of images.

    This script performs the following steps:
    1. Parses command-line arguments to configure the experiment.
    2. Loads and prepares fMRI data.
    3. Extracts CLIP or THINGSvision embeddings for the images.
    4. Optionally filters the embeddings and fMRI data to specific colors.
    5. Computes the Representational Dissimilarity Matrix (RDM) and visualizes it.
    6. Trains models using the extracted embeddings and RDM values.
    7. Plots training curves and evaluation metrics.

    Command-line Arguments:
        --root_dir (str): Path to the root data directory.
        --thingsvision (bool): Flag to use THINGSvision embeddings instead of CLIP.

    Outputs:
        - Visualizations of the RDM and training performance.
        - Trained models for predicting RDM values: both SVM (binary) and Siamese Network (contrastive learning)
        - Plots for all
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="CLIP + PyTorch Pipeline for RDM Modeling")
    parser.add_argument(
        "--root_dir", type=str, default=clip_config.ROOT_DATA_DIR, help="Path to the root data directory"
    )  # example local shortcut path (per README): "/Users/lindsayhexter/Library/CloudStorage/GoogleDrive-cinnabonswirl123@gmail.com/My Drive/Colab Notebooks"
    parser.add_argument(
        "--thingsvision", action="store_true", help="Enable the thingsvision flag. Defaults to False if not provided."
    )  # use THINGSvision embeddings rather than CLIP (assuming filepath is updated in clip_config to ...thingsvision_features.npy)
    try:
        args = parser.parse_args()
        if not os.path.exists(args.root_dir):  # noqa: PTH110
            raise FileNotFoundError(f"Root directory does not exist: {args.root_dir}")
    except argparse.ArgumentError as e:
        logger.error("Error parsing command-line arguments: %s", e)
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(1)

    root_data_dir = f"{args.root_dir}/{clip_config.ROOT_DATA_DIR}"

    #######################
    #   LOAD / PREP FMRI    STAGE <1>
    #######################
    fmri_data = prepare_fmri_data(
        root_data_dir=root_data_dir,  # we create the full directory path, so Shortcut path + data directory, e.g. "mini_data_for_python"
    )
    #######################
    # GET EMBEDDINGS    STAGE <2>
    #######################
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images = load_images(args.root_dir)
    embeddings = get_image_embeddings(
        images=images,
        device=device,
        is_thingsvision=args.thingsvision,
    )

    if clip_config.DISTRIBUTION_TYPE == "colors":
        if len(clip_config.COLOR_ARRAY_MAP_FILES) > 0:  # load color map from file
            color_mask_list = load_color_map_files(clip_config.COLOR_ARRAY_MAP_FILES, root_data_dir)
        else:  # load color map from scratch
            image_dir = f"{root_data_dir}/subj0{clip_config.SUBJECT}/training_split/training_images"
            images_to_classify = preprocess_images(
                image_dir,
                clip_config.DESIRED_IMAGE_NUMBER,
                clip_config.NEW_WIDTH,
                clip_config.NEW_HEIGHT,
                grayscale=False,
            )
            color_mask_list = classify_images_rgb(images_to_classify, threshold=0.70)
            print_image_counts(color_mask_list)
            # Sanity check - show images classified as blue
            display_images_by_label(images, color_mask_list, label_to_show=0, num_images=10)
        fmri_data, embeddings = get_equal_color_data(embeddings, fmri_data, color_mask_list, clip_config.COLOR_PAIR)
    #######################
    #   CREATE AND VISUALIZE RDM
    #######################
    rdm = create_rdm(fmri_data, metric=clip_config.METRIC)  # e.g. correlation, euclidean
    binary_rdm = create_binary_rdm(rdm, clip_config.METRIC)  # e.g. correlation, euclidean
    # Plot a subset of the RDM for visualization and values distribution of the full RDM
    plot_rdm_submatrix(rdm, subset_size=100)
    plot_rdm_distribution(rdm, bins=30, exclude_diagonal=True)
    plot_rdm_submatrix(binary_rdm, subset_size=100, binary=True)
    plot_rdm_distribution(binary_rdm, bins=30, exclude_diagonal=True)
    # RDM highest/lowest/closest to 1 values
    analyze_rdm(rdm, images)

    #######################
    #   TRAINING    STAGE <3>
    #######################

    run_svm(fmri_data, rdm, binary_rdm)

    run_kfold_cv(fmri_data, binary_rdm, rdm, device)


if __name__ == "__main__":
    main()
