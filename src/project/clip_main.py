# clip_main.py

"""clip_main Module

This module serves as the entry point for running the CLIP model implementation.

Functions:
    main(): The main function that starts the application.
"""

import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .config import clip_config
from .models.pytorch_models import DynamicLayerSizeNeuralNetwork, NeuralNetwork
from .utils.clip_utils import get_image_embeddings, load_clip_model
from .utils.data_utils import create_rdm, prepare_fmri_data
from .utils.pytorch_data import PairDataset, generate_pair_indices, train_test_split_pairs
from .utils.pytorch_training import reconstruct_predicted_rdm, train_model
from .utils.visualizations import (
    plot_accuracy_vs_layers,
    plot_rdm_distribution,
    plot_rdm_submatrix,
    plot_rdms,
    plot_training_history,
)


def main():
    #######################
    #   LOAD / PREP FMRI
    #######################
    fmri_data = prepare_fmri_data(
        subj=clip_config.SUBJECT,
        desired_image_number=clip_config.DESIRED_IMAGE_NUMBER,
        roi=clip_config.ROI,
        region_class=clip_config.REGION_CLASS,
        root_data_dir=clip_config.ROOT_DATA_DIR,
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
        clip_config.ROOT_DATA_DIR, f"subj0{clip_config.SUBJECT}", "training_split", "training_images"
    )

    embeddings = get_image_embeddings(
        image_dir=image_dir,
        processor=processor_clip,
        model=model_clip,
        desired_image_number=clip_config.DESIRED_IMAGE_NUMBER,
        device=device,
    )
    print("Embeddings shape:", embeddings.shape)

    #######################
    #    DATA PAIRS
    #######################
    row_indices, col_indices, rdm_values = generate_pair_indices(rdm)
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

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=clip_config.LEARNING_RATE)

    train_loss, train_acc, test_loss, test_acc = train_model(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        device,
        num_epochs=clip_config.EPOCHS,
        metric=clip_config.ACCURACY,
    )

    # Plot training curves
    plot_training_history(train_loss, train_acc, test_loss, test_acc, metric=clip_config.ACCURACY)

    #######################
    #   RDM RECONSTRUCTION
    #######################
    predicted_rdm = reconstruct_predicted_rdm(model, embeddings, row_indices, col_indices, device)
    plot_rdms(rdm, predicted_rdm)

    #######################
    #   LAYER SWEEP
    #######################
    if clip_config.SWEEP_LAYERS:
        layers_list = [0, 1, 2, 3]
        accuracy_list = []
        for layer_num in layers_list:
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

            # Optionally save
            save_path = f"dynamic_model_{layer_num}_layers.pth"
            torch.save(sweep_model.state_dict(), save_path)

        plot_accuracy_vs_layers(layers_list, accuracy_list, metric=clip_config.ACCURACY)


if __name__ == "__main__":
    main()
