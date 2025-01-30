# main.py

"""main Module

This module serves as the entry point for running the CLIP model implementation.

Functions:
    main(): The main function that starts the application.
"""

import argparse
import os
import sys

import numpy as np
import torch

from ..project.logger import logger
from .config import clip_config
from .utils.clip_utils import get_image_embeddings, load_images
from .utils.data_utils import analyze_rdm, create_rdm, prepare_fmri_data
from .utils.pytorch_data import generate_pair_indices
from .utils.pytorch_training import train_all
from .utils.visualizations import all_plots, plot_accuracy_vs_layers, plot_rdm_distribution, plot_rdm_submatrix


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="CLIP + PyTorch Pipeline for RDM Modeling")
    parser.add_argument(
        "--root_dir", type=str, default=clip_config.ROOT_DATA_DIR, help="Path to the root data directory"
    )  # example local shortcut path (per README): "/Users/lindsayhexter/Library/CloudStorage/GoogleDrive-cinnabonswirl123@gmail.com/My Drive/Colab Notebooks"
    parser.add_argument(
        "--thingsvision", action="store_true", help="Enable the thingsvision flag. Defaults to False if not provided."
    )
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

    #######################
    #   LOAD / PREP FMRI
    #######################
    fmri_data = prepare_fmri_data(
        root_data_dir=f"{args.root_dir}/{clip_config.ROOT_DATA_DIR}",  # we create the full directory path, so Shortcut path + data directory, e.g. "mini_data_for_python"
    )
    # CREATE AND VISUALIZE RDM
    rdm = create_rdm(fmri_data, metric=clip_config.METRIC)  # e.g. correlation, euclidean
    # Plot subset + distribution
    plot_rdm_submatrix(rdm, subset_size=100)
    plot_rdm_distribution(rdm, bins=30, exclude_diagonal=True)

    # Load all images
    images = load_images(args.root_dir)

    # RDM highest/lowest/closest to 1 values
    analyze_rdm(rdm, images)

    #######################
    #     CLIP EMBEDDINGS
    #######################
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = get_image_embeddings(
        images=images,
        device=device,
        is_thingsvision=args.thingsvision,
    )

    #######################
    #      START TRAINING
    row_indices, col_indices, rdm_values = generate_pair_indices(rdm)
    train_loss, train_acc, test_loss, test_acc = train_all(row_indices, col_indices, rdm_values, embeddings, device)

    #######################
    #    PLOT TRAINING CURVES
    #    NOTE: For KFOLD - we plot over N Folds (vs Standard training, over N Epochs)
    #######################
    all_plots(train_loss, train_acc, test_loss, test_acc)

    #######################
    #   LAYER SWEEP
    #######################
    if clip_config.SWEEP_LAYERS:  # True or False - if True, sweep over a list of layers
        accuracy_list = []
        for layer_num in clip_config.LAYERS_LIST:  # e.g. [0, 1, 2, 3]
            logger.info("Starting layer sweep for %s hidden layers.", layer_num)
            _, _, _, sweep_test_acc = train_all(
                row_indices, col_indices, rdm_values, embeddings, device, hidden_layers=layer_num
            )
            if not clip_config.K_FOLD:
                accuracy_list.append(max(sweep_test_acc))
            else:
                accuracy_list.append(np.mean(sweep_test_acc))

        plot_accuracy_vs_layers(
            clip_config.LAYERS_LIST, accuracy_list, is_thingsvision=args.thingsvision, metric=clip_config.ACCURACY
        )


if __name__ == "__main__":
    main()
