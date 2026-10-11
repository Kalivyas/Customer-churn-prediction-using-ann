"""
train.py
Trains two models on Churn_Modelling.csv:
  1. Baseline ANN  - imbalance handled via Keras class_weight only
  2. Proposed ANN  - imbalance handled via SMOTE oversampling of the
                     training set (CO5 innovation), same architecture,
                     same hyperparameters, same random seed.

Run:
    python src/train.py --data data/Churn_Modelling.csv --out models
"""

import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE

sys.path.append(os.path.dirname(__file__))
from preprocessing import load_and_prepare, get_class_weight, RANDOM_STATE
from model import build_ann, get_early_stopping

EPOCHS = 150
BATCH_SIZE = 32


def set_seeds():
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)


def train_baseline(X_train, y_train, n_features):
    set_seeds()
    model = build_ann(n_features)
    class_weight = get_class_weight(y_train)
    early_stop = get_early_stopping()
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=0,
    )
    return model, history.history


def train_proposed_smote(X_train, y_train, n_features):
    set_seeds()
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"  SMOTE resampled train set: {len(y_train)} -> {len(y_res)} "
          f"(class balance now {np.bincount(y_res) / len(y_res)})")

    model = build_ann(n_features)
    early_stop = get_early_stopping()
    history = model.fit(
        X_res, y_res,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=0,
    )
    return model, history.history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/Churn_Modelling.csv")
    parser.add_argument("--out", default="models")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, feature_cols, scaler = load_and_prepare(args.data)
    n_features = X_train.shape[1]
    print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape} | Features: {feature_cols}")

    print("\n=== Training Baseline ANN (class_weight only) ===")
    baseline_model, baseline_hist = train_baseline(X_train, y_train, n_features)
    print(f"  Baseline trained for {len(baseline_hist['loss'])} epochs (early stopping)")

    print("\n=== Training Proposed ANN (SMOTE oversampling) ===")
    proposed_model, proposed_hist = train_proposed_smote(X_train, y_train, n_features)
    print(f"  Proposed trained for {len(proposed_hist['loss'])} epochs (early stopping)")

    baseline_model.save(os.path.join(args.out, "baseline_ann.keras"))
    proposed_model.save(os.path.join(args.out, "proposed_ann_smote.keras"))

    with open("results/baseline_history.json", "w") as f:
        json.dump(baseline_hist, f)
    with open("results/proposed_history.json", "w") as f:
        json.dump(proposed_hist, f)

    # Save the split test set + scaler params so test.py evaluates on
    # the EXACT same held-out data (identical random_state as here).
    np.savez(os.path.join(args.out, "test_split.npz"),
             X_test=X_test, y_test=y_test, X_train=X_train, y_train=y_train)

    print("\nTraining complete. Models saved to:", args.out)


if __name__ == "__main__":
    main()
