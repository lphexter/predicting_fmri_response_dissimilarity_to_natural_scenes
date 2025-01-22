# main.py

"""main Module

This module serves as the entry point for running the CLIP model implementation.

Functions:
    main(): The main function that starts the application.
"""

import argparse
import os

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .config import clip_config
from .models.pytorch_models import DynamicLayerSizeNeuralNetwork, NeuralNetwork
from .utils.clip_utils import get_image_embeddings, load_clip_model
from .utils.data_utils import create_rdm, prepare_fmri_data
from .utils.pytorch_data import PairDataset, generate_pair_indices, prepare_data_for_kfold, train_test_split_pairs
from .utils.pytorch_training import reconstruct_predicted_rdm, train_model, train_model_kfold
from .utils.visualizations import (
    plot_accuracy_vs_layers,
    plot_rdm_distribution,
    plot_rdm_submatrix,
    plot_rdms,
    plot_training_history,
)


def main():  # noqa: PLR0915
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="CLIP + PyTorch Pipeline for RDM Modeling")
    parser.add_argument(
        "--root_dir", type=str, default=clip_config.ROOT_DATA_DIR, help="Path to the root data directory"
    )
    args = parser.parse_args()

    #######################
    #   LOAD / PREP FMRI
    #######################
    fmri_data = prepare_fmri_data(
        subj=clip_config.SUBJECT,
        desired_image_number=clip_config.DESIRED_IMAGE_NUMBER,
        roi=clip_config.ROI,
        region_class=clip_config.REGION_CLASS,
        root_data_dir=f"{args.root_dir}/{clip_config.ROOT_DATA_DIR}",
    )
    rdm = create_rdm(fmri_data, metric=clip_config.METRIC)

    # Plot subset + distribution
    plot_rdm_submatrix(rdm, subset_size=100)
    plot_rdm_distribution(rdm, bins=30, exclude_diagonal=True)

    #######################
    #     CLIP EMBEDDINGS
    #######################
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_clip, processor_clip = load_clip_model(pretrained_model_name=clip_config.PRETRAINED_MODEL, device=device)

    image_dir = os.path.join(
        f"{args.root_dir}/{clip_config.ROOT_DATA_DIR}",
        f"subj0{clip_config.SUBJECT}",
        "training_split",
        "training_images",
    )

    embeddings = get_image_embeddings(
        image_dir=image_dir,
        processor=processor_clip,
        model=model_clip,
        desired_image_number=clip_config.DESIRED_IMAGE_NUMBER,
        device=device,
    )
    print("Embeddings shape:", embeddings.shape)

    row_indices, col_indices, rdm_values = generate_pair_indices(rdm)
    criterion = nn.MSELoss()

    if not clip_config.K_FOLD:  # standard training
        #######################
        #    DATA PAIRS
        #######################

        X_train_indices, X_test_indices, y_train, y_test = train_test_split_pairs(
            row_indices, col_indices, rdm_values, test_size=clip_config.TEST_SIZE
        )

        train_dataset = PairDataset(embeddings, X_train_indices, y_train)
        test_dataset = PairDataset(embeddings, X_test_indices, y_test)

        train_loader = DataLoader(train_dataset, batch_size=clip_config.BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=clip_config.BATCH_SIZE, shuffle=False)

        #######################
        #   MODEL + TRAIN
        #######################
        model = NeuralNetwork(hidden_layers=clip_config.HIDDEN_LAYERS, activation_func=clip_config.ACTIVATION_FUNC).to(
            device
        )

        optimizer = optim.Adam(model.parameters(), lr=clip_config.LEARNING_RATE)
        train_loss, train_acc, test_loss, test_acc = train_model(
            model,
            train_loader,
            test_loader,
            criterion,
            optimizer,
            device,
            num_epochs=clip_config.EPOCHS,
        )

    else:
        loaders = prepare_data_for_kfold(embeddings, rdm, n_splits=clip_config.K_FOLD_SPLITS)
        train_loss, train_acc, test_loss, test_acc = train_model_kfold(
            loaders, criterion, device, num_layers=clip_config.HIDDEN_LAYERS, num_epochs=clip_config.EPOCHS
        )

    #######################
    #    PLOT TRAINING CURVES
    #######################
    if not clip_config.K_FOLD:
        # Standard training mode plotting
        plot_training_history(train_loss, train_acc, test_loss, test_acc, metric=clip_config.ACCURACY)
    else:
        # K-Fold mode: Calculate mean and std across folds
        train_loss_mean = np.mean(train_loss, axis=0)
        train_acc_mean = np.mean(train_acc, axis=0)
        test_loss_mean = np.mean(test_loss, axis=0)
        test_acc_mean = np.mean(test_acc, axis=0)

        train_loss_std = np.std(train_loss, axis=0)
        train_acc_std = np.std(train_acc, axis=0)
        test_loss_std = np.std(test_loss, axis=0)
        test_acc_std = np.std(test_acc, axis=0)

        plot_training_history(
            train_loss_mean,
            train_acc_mean,
            test_loss_mean,
            test_acc_mean,
            metric=clip_config.ACCURACY,
            std_dev=True,
            train_loss_std=train_loss_std,
            train_acc_std=train_acc_std,
            test_loss_std=test_loss_std,
            test_acc_std=test_acc_std,
        )

    #######################
    #   RDM RECONSTRUCTION
    #######################
    if not clip_config.K_FOLD:  # not supported for K-FOLD
        predicted_rdm = reconstruct_predicted_rdm(model, embeddings, row_indices, col_indices, device)
        plot_rdms(rdm, predicted_rdm)
    else:
        print("RDM Reconstruction is not implemented for K-Fold mode.")

    #######################
    #   LAYER SWEEP
    #######################
    if clip_config.SWEEP_LAYERS:
        accuracy_list = []
        for layer_num in clip_config.LAYERS_LIST:
            print(f"\nStarting layer sweep for {layer_num} hidden layers.")
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
                    metric=clip_config.ACCURACY,
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
