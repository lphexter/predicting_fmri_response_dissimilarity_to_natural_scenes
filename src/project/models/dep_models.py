# models/models.py

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers


def create_cnn_model(input_shape, activation_func="sigmoid"):
    """Creates a simple Siamese CNN model with two input branches."""
    input_1 = keras.Input(shape=input_shape, name="input_image_A")
    input_2 = keras.Input(shape=input_shape, name="input_image_B")

    def conv_block(x):
        x = layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        return layers.Flatten()(x)

    processed_1 = conv_block(input_1)
    processed_2 = conv_block(input_2)

    merged = layers.concatenate([processed_1, processed_2])

    if activation_func == "linear":
        output = layers.Dense(1, activation="linear")(merged)
    elif activation_func == "sigmoid":
        out_layer = layers.Dense(1, activation="sigmoid")(merged)
        output = out_layer * 2.0
    else:
        raise ValueError(f"Invalid activation function: {activation_func}")

    return keras.Model(inputs=[input_1, input_2], outputs=output, name="SiameseCNNModel")


def correlation_loss_with_mse(y_true, y_pred, alpha=0.5):
    """Combined loss encouraging high correlation and penalizing numeric differences.

    alpha=1 => correlation-only, alpha=0 => MSE-only.
    """
    # 1) Pearson correlation (negative sign to maximize correlation)
    r = tfp.stats.correlation(y_true, y_pred, sample_axis=0, event_axis=None)
    correlation_loss_val = -r  # Negative sign to maximize correlation

    # 2) MSE
    mse_loss_val = tf.reduce_mean(tf.square(y_true - y_pred))

    # 3) Weighted sum
    return alpha * correlation_loss_val + (1 - alpha) * mse_loss_val  # the loss
