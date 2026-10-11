"""
cross_validation.py
5-fold Stratified Cross-Validation (k=5) for the Baseline ANN
(class_weight imbalance handling) on the training split.

Run:
    python src/cross_validation.py --data data/Churn_Modelling.csv --out results
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score)

sys.path.append(os.path.dirname(__file__))
from preprocessing import load_and_prepare, get_class_weight, RANDOM_STATE
from model import build_ann, get_early_stopping

K = 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/Churn_Modelling.csv")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    X_train, X_test, y_train, y_test, feature_cols, scaler = load_and_prepare(args.data)
    n_features = X_train.shape[1]

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        np.random.seed(RANDOM_STATE)
        tf.random.set_seed(RANDOM_STATE)
        m = build_ann(n_features)
        m.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=100, batch_size=32,
              class_weight=get_class_weight(y_tr), callbacks=[get_early_stopping()], verbose=0)

        proba = m.predict(X_va, verbose=0).ravel()
        pred = (proba >= 0.5).astype(int)
        row = dict(fold=fold,
                   accuracy=accuracy_score(y_va, pred),
                   precision=precision_score(y_va, pred),
                   recall=recall_score(y_va, pred),
                   f1=f1_score(y_va, pred),
                   roc_auc=roc_auc_score(y_va, proba))
        rows.append(row)
        print(f"Fold {fold}: " + "  ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "fold"))

    df = pd.DataFrame(rows)
    mean = df.drop(columns="fold").mean()
    std = df.drop(columns="fold").std(ddof=0)
    print("\nMean +/- Std over", K, "folds")
    for c in mean.index:
        print(f"  {c:9s}: {mean[c]:.4f} +/- {std[c]:.4f}")

    out_df = df.copy()
    out_df["fold"] = out_df["fold"].astype(str)
    out_df.loc[len(out_df)] = ["mean"] + mean.tolist()
    out_df.loc[len(out_df)] = ["std"] + std.tolist()
    out_df.round(4).to_csv(os.path.join(args.out, "cv_results.csv"), index=False)

    plt.figure(figsize=(6.5, 4.8))
    plt.bar(df["fold"], df["roc_auc"], color="#2563eb", alpha=0.85)
    plt.axhline(mean["roc_auc"], color="red", linestyle="--", label=f"Mean AUC = {mean['roc_auc']:.3f}")
    plt.xlabel("Fold"); plt.ylabel("ROC-AUC")
    plt.title(f"{K}-Fold Stratified CV — ROC-AUC per Fold (Baseline ANN)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out, "cv_results.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
