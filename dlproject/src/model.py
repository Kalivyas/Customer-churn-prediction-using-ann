"""
model.py
Defines the ANN architecture used for both the Baseline and the
SMOTE-based Proposed model (CO5 innovation). Same architecture is kept
constant across experiments so that any performance difference is
attributable to the imbalance-handling strategy, not the network itself.
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_ann(input_dim, regularizer=None, l_rate=0.001):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(16, activation="relu", kernel_regularizer=regularizer),
        layers.Dense(8, activation="relu", kernel_regularizer=regularizer),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=l_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def get_early_stopping(patience=10):
    return keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )
