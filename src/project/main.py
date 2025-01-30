# main.py

"""main Module

This module serves as the entry point for running the CLIP model implementation.

Functions:
    main(): The main function that starts the application.
"""

import argparse
import logging
import os

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .config import clip_config
from .models.pytorch_models import DynamicLayerSizeNeuralNetwork
from .utils.clip_utils import get_image_embeddings, load_images
from .utils.data_utils import analyze_rdm, compare_rdms, create_rdm, prepare_fmri_data
from .utils.pytorch_data import PairDataset, generate_pair_indices, prepare_data_for_kfold, train_test_split_pairs
from .utils.pytorch_training import reconstruct_predicted_rdm, train_model, train_model_kfold
from .utils.visualizations import (
    plot_accuracy_vs_layers,
    plot_rdm_distribution,
    plot_rdm_submatrix,
    plot_rdms,
    plot_training_history,
    show_image_pair,
)


def main():  # noqa: PLR0915
    # initialize logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="CLIP + PyTorch Pipeline for RDM Modeling")
    parser.add_argument(
        "--root_dir", type=str, default=clip_config.ROOT_DATA_DIR, help="Path to the root data directory"
    )  # example local shortcut path (per README): "/Users/lindsayhexter/Library/CloudStorage/GoogleDrive-cinnabonswirl123@gmail.com/My Drive/Colab Notebooks"
    parser.add_argument(
        "--thingsvision", action="store_true", help="Enable the thingsvision flag. Defaults to False if not provided."
    )
    args = parser.parse_args()

    #######################
    #   LOAD / PREP FMRI
    #######################
    fmri_data = prepare_fmri_data(
        subj=clip_config.SUBJECT,  # e.g. "1"
        desired_image_number=clip_config.DESIRED_IMAGE_NUMBER,  # e.g. "50"
        roi=clip_config.ROI,  # e.g. "V1v"
        region_class=clip_config.REGION_CLASS,  # e.g. "Visual"
        root_data_dir=f"{args.root_dir}/{clip_config.ROOT_DATA_DIR}",  # we create the full directory path, so Shortcut path + data directory, e.g. "mini_data_for_python"
    )
    logging.info("fMRI data shape: %s", fmri_data.shape)
    rdm = create_rdm(fmri_data, metric=clip_config.METRIC)  # e.g. correlation, euclidean

    # Plot subset + distribution
    plot_rdm_submatrix(rdm, subset_size=100)
    plot_rdm_distribution(rdm, bins=30, exclude_diagonal=True)

    #######################
    #     CLIP EMBEDDINGS
    #######################
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_dir = os.path.join(
        f"{args.root_dir}/{clip_config.ROOT_DATA_DIR}",
        f"subj0{clip_config.SUBJECT}",
        "training_split",
        "training_images",
    )
    images = load_images(image_dir=image_dir, desired_image_number=clip_config.DESIRED_IMAGE_NUMBER)
    embeddings = get_image_embeddings(
        images=images,
        desired_image_number=clip_config.DESIRED_IMAGE_NUMBER,
        device=device,
        is_thingsvision=args.thingsvision,
    )
    logging.info("Embeddings shape: %s", embeddings.shape)

    #######################
    #     RDM HIGHEST/LOWEST/CLOSEST_TO_1 VALUES
    #######################

    results = analyze_rdm(rdm, clip_config.METRIC)
    logging.info("RDM Value Analysis Results: %s", results)

    for key in results:
        title = f"Pair images of {key} value with score {results[key]['value']}"
        image_pair = results[key]["pair"]
        show_image_pair(image_pair[0], image_pair[1], images, title)

    #######################

    row_indices, col_indices, rdm_values = generate_pair_indices(rdm)
    criterion = nn.MSELoss()

    if not clip_config.K_FOLD:  # standard training
        #######################
        #    DATA PAIRS
        #######################
        logging.info("Starting standard training (not KFold)")

        X_train_indices, X_test_indices, y_train, y_test = train_test_split_pairs(
            row_indices,
            col_indices,
            rdm_values,
            test_size=clip_config.TEST_SIZE,  # e.g. 0.2 = 20% test
        )

        train_dataset = PairDataset(embeddings, X_train_indices, y_train)
        test_dataset = PairDataset(embeddings, X_test_indices, y_test)

        train_loader = DataLoader(
            train_dataset, batch_size=clip_config.BATCH_SIZE, shuffle=True
        )  # e.g. batch size of 32
        test_loader = DataLoader(test_dataset, batch_size=clip_config.BATCH_SIZE, shuffle=False)

        #######################
        #   MODEL + TRAIN
        #######################
        model = DynamicLayerSizeNeuralNetwork(
            hidden_layers=clip_config.HIDDEN_LAYERS,
            activation_func=clip_config.ACTIVATION_FUNC,  # e.g. sigmoid, linear
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=clip_config.LEARNING_RATE)  # e.g. 0.1
        train_loss, train_acc, test_loss, test_acc = train_model(
            model,
            train_loader,
            test_loader,
            criterion,
            optimizer,
            device,
            num_epochs=clip_config.EPOCHS,  # e.g. 5 epochs
        )

    else:
        logging.info("Starting KFold Training")
        loaders = prepare_data_for_kfold(
            row_indices,
            col_indices,
            rdm_values,
            embeddings,
            n_splits=clip_config.K_FOLD_SPLITS,  # e.g. 5
        )
        train_loss, train_acc, test_loss, test_acc = train_model_kfold(
            loaders,
            criterion,
            device,
            num_layers=clip_config.HIDDEN_LAYERS,  # e.g. 1 hidden layer
            num_epochs=clip_config.EPOCHS,
        )

    #######################
    #    PLOT TRAINING CURVES
    #       NOTE: for standard KFold, we take in a list - but standard training we input a list over N epochs, vs in KFold, we input a list over N splits
    #######################
    if not clip_config.K_FOLD:
        logging.info("Standard historical plotting over training course (not KFold)")
        # Standard training mode plotting
        plot_training_history(
            train_loss, train_acc, test_loss, test_acc, metric=clip_config.ACCURACY
        )  # e.g. r2, spearman, pearson
    else:
        logging.info("KFold historical plotting over training course with stdev")
        train_loss_std = np.std(train_loss, axis=0)
        train_acc_std = np.std(train_acc, axis=0)
        test_loss_std = np.std(test_loss, axis=0)
        test_acc_std = np.std(test_acc, axis=0)

        plot_training_history(
            train_loss,
            train_acc,
            test_loss,
            test_acc,
            metric=clip_config.ACCURACY,
            std_dev=True,
            train_loss_std=train_loss_std,
            train_acc_std=train_acc_std,
            test_loss_std=test_loss_std,
            test_acc_std=test_acc_std,
        )

    ##################################
    #   RDM RECONSTRUCTION OF TEST RDMs
    ##################################
    if not clip_config.K_FOLD:  # not supported for K-FOLD
        predicted_rdm = reconstruct_predicted_rdm(model, embeddings, row_indices, col_indices, device)
        plot_rdms(rdm, predicted_rdm)
        compare_rdms(rdm, predicted_rdm)
    else:
        logging.info("RDM Reconstruction is not implemented for K-Fold mode.")

    #######################
    #   LAYER SWEEP
    #######################
    if clip_config.SWEEP_LAYERS:  # True or False - if True, sweep over a list of layers
        accuracy_list = []
        for layer_num in clip_config.LAYERS_LIST:  # e.g. [0, 1, 2, 3]
            logging.info("\nStarting layer sweep for %s hidden layers.", layer_num)
            if not clip_config.K_FOLD:
                sweep_model = DynamicLayerSizeNeuralNetwork(
                    hidden_layers=layer_num, activation_func=clip_config.ACTIVATION_FUNC
                ).to(device)

                sweep_optimizer = optim.Adam(sweep_model.parameters(), lr=clip_config.LEARNING_RATE)
                sweep_train_loss, sweep_train_acc, sweep_test_loss, sweep_test_acc = train_model(
                    sweep_model,
                    train_loader,
                    test_loader,
                    criterion,
                    sweep_optimizer,
                    device,
                    num_epochs=clip_config.EPOCHS,
                )
                accuracy_list.append(max(sweep_test_acc))
                # Optionally save - not implemented for K-FOLD mode
                # save_path = f"dynamic_model_{layer_num}_layers.pth"  # noqa: ERA001
                # torch.save(sweep_model.state_dict(), save_path)  # noqa: ERA001
            else:
                sweep_train_loss, sweep_train_acc, sweep_test_loss, sweep_test_acc = train_model_kfold(
                    loaders, criterion, device, num_layers=layer_num, num_epochs=clip_config.EPOCHS
                )
                accuracy_list.append(np.mean(sweep_test_acc))

        plot_accuracy_vs_layers(clip_config.LAYERS_LIST, accuracy_list, metric=clip_config.ACCURACY)


if __name__ == "__main__":
    main()
