# dep_main.py ---> code to run deprecated first model

import argparse
import os

import tensorflow as tf
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from tensorflow.keras.callbacks import EarlyStopping

from ..project.logger import logger

# Local imports
from .config import dep_config
from .models.dep_models import correlation_loss_with_mse, create_cnn_model
from .utils.data_utils import (
    create_rdm,
    create_rdm_from_vectors,
    data_generator,
    prepare_data_for_cnn,
    prepare_fmri_data,
    preprocess_images,
)
from .utils.visualizations import (
    plot_rdm_distribution,
    plot_rdm_submatrix,
    plot_rdms,
)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Keras Pipeline for RDM Modeling")
    parser.add_argument(
        "--root_dir", type=str, default=dep_config.ROOT_DATA_DIR, help="Path to the root data directory"
    )
    args = parser.parse_args()

    ########################
    #   PREPARE FMRI DATA
    ########################
    fmri_data = prepare_fmri_data(
        subj=dep_config.SUBJECT,
        desired_image_number=dep_config.DESIRED_IMAGE_NUMBER,
        roi=dep_config.ROI,
        region_class=dep_config.REGION_CLASS,
        root_data_dir=f"{args.root_dir}/{dep_config.ROOT_DATA_DIR}",
    )

    ########################
    #   PREPROCESS IMAGES
    ########################
    image_dir = os.path.join(
        f"{args.root_dir}/{dep_config.ROOT_DATA_DIR}", f"subj0{dep_config.SUBJECT}", "training_split", "training_images"
    )
    images = preprocess_images(
        image_dir=image_dir,
        num_images=dep_config.DESIRED_IMAGE_NUMBER,
        new_width=dep_config.NEW_WIDTH,
        new_height=dep_config.NEW_HEIGHT,
    )
    logger.info(f"Images shape: {images.shape}")

    ########################
    #   CREATE & ANALYZE RDM
    ########################
    rdm = create_rdm(fmri_data, metric=dep_config.METRIC)
    plot_rdm_submatrix(rdm, subset_size=100)
    plot_rdm_distribution(rdm, bins=30, exclude_diagonal=True)

    ########################
    #   PREPARE FOR TRAINING
    ########################
    X_train_indices, X_test_indices, y_train, y_test = prepare_data_for_cnn(rdm, test_size=dep_config.TEST_SIZE)

    train_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(images, X_train_indices, y_train, batch_size=dep_config.BATCH_SIZE),
        output_signature=(
            (
                tf.TensorSpec(
                    shape=(None, dep_config.NEW_HEIGHT, dep_config.NEW_WIDTH, dep_config.NUM_CHANNELS), dtype=tf.float32
                ),
                tf.TensorSpec(
                    shape=(None, dep_config.NEW_HEIGHT, dep_config.NEW_WIDTH, dep_config.NUM_CHANNELS), dtype=tf.float32
                ),
            ),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
        ),
    )

    test_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(images, X_test_indices, y_test, batch_size=dep_config.BATCH_SIZE),
        output_signature=(
            (
                tf.TensorSpec(
                    shape=(None, dep_config.NEW_HEIGHT, dep_config.NEW_WIDTH, dep_config.NUM_CHANNELS), dtype=tf.float32
                ),
                tf.TensorSpec(
                    shape=(None, dep_config.NEW_HEIGHT, dep_config.NEW_WIDTH, dep_config.NUM_CHANNELS), dtype=tf.float32
                ),
            ),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
        ),
    )

    ########################
    #   CREATE & COMPILE MODEL
    ########################
    model = create_cnn_model(
        input_shape=(dep_config.NEW_HEIGHT, dep_config.NEW_WIDTH, dep_config.NUM_CHANNELS),
        activation_func=dep_config.ACTIVATION_FUNC,
    )
    model.compile(
        loss=lambda y_true, y_pred: correlation_loss_with_mse(y_true, y_pred, alpha=dep_config.ALPHA), optimizer="adam"
    )
    model.summary()

    ########################
    #       TRAIN
    ########################
    model.fit(
        train_dataset,
        steps_per_epoch=len(y_train) // dep_config.BATCH_SIZE,
        epochs=dep_config.EPOCHS,
        validation_data=test_dataset,
        validation_steps=len(y_test) // dep_config.BATCH_SIZE,
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
    )

    ########################
    #     EVALUATE
    ########################
    y_pred = model.predict(test_dataset, steps=(len(y_test) // dep_config.BATCH_SIZE) + 1).flatten()

    # Evaluate based on chosen metric
    if dep_config.ACCURACY == "spearman":
        corr_val, _ = spearmanr(y_pred, y_test)
        logger.info(f"Spearman Correlation: {corr_val}")
    elif dep_config.ACCURACY == "pearson":
        corr_val, _ = pearsonr(y_pred, y_test)
        logger.info(f"Pearson Correlation: {corr_val}")
    elif dep_config.ACCURACY == "r2":
        r2_val = r2_score(y_test, y_pred)
        logger.info(f"R^2 Score: {r2_val}")
    else:
        raise ValueError(f"Invalid accuracy metric: {dep_config.ACCURACY}")

    # Log all three, for reference
    corr_sp, _ = spearmanr(y_pred, y_test)
    corr_pe, _ = pearsonr(y_pred, y_test)
    r2_val = r2_score(y_test, y_pred)
    logger.info(f"[All metrics] Spearman: {corr_sp}, Pearson: {corr_pe}, R^2: {r2_val}")

    ########################
    #     RDM COMPARISON
    ########################
    # Just an example of partially reconstructing:
    predicted_rdm = create_rdm_from_vectors(y_pred[:1000])
    ground_truth_rdm = create_rdm_from_vectors(y_test[:1000])
    plot_rdms(ground_truth_rdm, predicted_rdm)


if __name__ == "__main__":
    main()
