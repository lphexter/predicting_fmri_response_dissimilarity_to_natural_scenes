# main.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score

# Local imports
import config
from data_utils import (
    prepare_fmri_data,
    preprocess_images,
    create_rdm,
    create_rdm_from_vectors,
)
from models import create_cnn_model, correlation_loss_with_mse
from visualizations import (
    plot_rdm_submatrix,
    plot_rdm_distribution,
    plot_rdms,
)


def prepare_data_for_cnn(rdm, test_size=0.2):
    """
    Splits the upper-tri pairs of RDM into train/test sets.
    Returns pair indices and corresponding RDM values for train/test.
    """
    from sklearn.model_selection import train_test_split
    num_images = rdm.shape[0]
    row_indices, col_indices = np.triu_indices(num_images, k=1)
    rdm_values = rdm[row_indices, col_indices]

    train_indices, test_indices, y_train, y_test = train_test_split(
        np.arange(len(rdm_values)), rdm_values, test_size=test_size, random_state=42
    )

    X_train_indices = (row_indices[train_indices], col_indices[train_indices])
    X_test_indices  = (row_indices[test_indices],  col_indices[test_indices])
    return X_train_indices, X_test_indices, y_train, y_test


def data_generator(image_data, pair_indices, y_data, batch_size=32):
    """
    Generator yielding (batch_x1, batch_x2), batch_y for Siamese model.
    """
    num_samples = len(y_data)
    row_indices, col_indices = pair_indices

    while True:
        for offset in range(0, num_samples, batch_size):
            end = offset + batch_size

            batch_rows = row_indices[offset:end]
            batch_cols = col_indices[offset:end]
            batch_x1 = image_data[batch_rows]
            batch_x2 = image_data[batch_cols]
            batch_y  = y_data[offset:end]

            batch_x1 = batch_x1[..., np.newaxis]
            batch_x2 = batch_x2[..., np.newaxis]

            yield (batch_x1, batch_x2), batch_y


def main():
    ########################
    #   PREPARE FMRI DATA
    ########################
    fmri_data = prepare_fmri_data(
        subj=config.SUBJECT,
        desired_image_number=config.DESIRED_IMAGE_NUMBER,
        roi=config.ROI,
        region_class=config.REGION_CLASS,
        root_data_dir=config.ROOT_DATA_DIR,
    )

    ########################
    #   PREPROCESS IMAGES
    ########################
    image_dir = os.path.join(
        config.ROOT_DATA_DIR, f"subj0{config.SUBJECT}",
        "training_split", "training_images"
    )
    images = preprocess_images(
        image_dir=image_dir,
        num_images=config.DESIRED_IMAGE_NUMBER,
        new_width=config.NEW_WIDTH,
        new_height=config.NEW_HEIGHT
    )
    print("Images shape:", images.shape)

    ########################
    #   CREATE & ANALYZE RDM
    ########################
    rdm = create_rdm(fmri_data, metric=config.METRIC)
    plot_rdm_submatrix(rdm, subset_size=100)
    plot_rdm_distribution(rdm, bins=30, exclude_diagonal=True)

    ########################
    #   PREPARE FOR TRAINING
    ########################
    X_train_indices, X_test_indices, y_train, y_test = prepare_data_for_cnn(
        rdm,
        test_size=config.TEST_SIZE
    )

    # Create tf.data.Dataset from generator
    train_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(images, X_train_indices, y_train, batch_size=config.BATCH_SIZE),
        output_signature=(
            (
                tf.TensorSpec(shape=(None, config.NEW_HEIGHT, config.NEW_WIDTH, config.NUM_CHANNELS), dtype=tf.float32),
                tf.TensorSpec(shape=(None, config.NEW_HEIGHT, config.NEW_WIDTH, config.NUM_CHANNELS), dtype=tf.float32),
            ),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
        ),
    )

    test_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(images, X_test_indices, y_test, batch_size=config.BATCH_SIZE),
        output_signature=(
            (
                tf.TensorSpec(shape=(None, config.NEW_HEIGHT, config.NEW_WIDTH, config.NUM_CHANNELS), dtype=tf.float32),
                tf.TensorSpec(shape=(None, config.NEW_HEIGHT, config.NEW_WIDTH, config.NUM_CHANNELS), dtype=tf.float32),
            ),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
        ),
    )

    ########################
    #   CREATE & COMPILE MODEL
    ########################
    model = create_cnn_model(
        input_shape=(config.NEW_HEIGHT, config.NEW_WIDTH, config.NUM_CHANNELS),
        activation_func=config.ACTIVATION_FUNC
    )
    model.compile(
        loss=lambda y_true, y_pred: correlation_loss_with_mse(y_true, y_pred, alpha=config.ALPHA),
        optimizer="adam"
    )
    model.summary()

    ########################
    #       TRAIN
    ########################
    model.fit(
        train_dataset,
        steps_per_epoch=len(y_train) // config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=test_dataset,
        validation_steps=len(y_test) // config.BATCH_SIZE,
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
    )

    ########################
    #     EVALUATE
    ########################
    y_pred = model.predict(test_dataset, steps=(len(y_test) // config.BATCH_SIZE) + 1).flatten()

    if config.ACCURACY == 'spearman':
        corr_val, _ = spearmanr(y_pred, y_test)
        print(f"Spearman Correlation: {corr_val}")
    elif config.ACCURACY == 'pearson':
        corr_val, _ = pearsonr(y_pred, y_test)
        print(f"Pearson Correlation: {corr_val}")
    elif config.ACCURACY == 'r2':
        r2_val = r2_score(y_test, y_pred)
        print(f"R^2 Score: {r2_val}")
    else:
        raise ValueError(f"Invalid accuracy metric: {config.ACCURACY}")

    # Also compute all three if you wish
    corr_sp, _ = spearmanr(y_pred, y_test)
    corr_pe, _ = pearsonr(y_pred, y_test)
    r2_val = r2_score(y_test, y_pred)
    print(f"[All metrics] Spearman: {corr_sp}, Pearson: {corr_pe}, R^2: {r2_val}")

    ########################
    #     RDM COMPARISON
    ########################
    predicted_rdm = create_rdm_from_vectors(y_pred[:1000])  # partial for demonstration
    rdm_true       = create_rdm_from_vectors(y_test[:1000])

    plot_rdms(rdm_true, predicted_rdm)

if __name__ == "__main__":
    main()
