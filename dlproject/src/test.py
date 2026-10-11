"""
test.py
Loads the trained Baseline and Proposed (SMOTE) models and the held-out
test split saved by train.py, then produces:
  - results/results.csv                 (metric table, both models)
  - results/confusion_matrix_baseline.png
  - results/confusion_matrix_proposed.png
  - results/roc_curve_comparison.png
  - results/training_curves_baseline.png
  - results/training_curves_proposed.png

Run:
    python src/test.py --models models --out results
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, accuracy_score,
    precision_score, recall_score, f1_score, ConfusionMatrixDisplay,
    classification_report
)


def evaluate_model(model, X_test, y_test, threshold=0.5, label="Model"):
    y_proba = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_proba >= threshold).astype(int)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    print(f"\n--- {label} Test Set Metrics ---")
    print(f"Accuracy   : {acc:.4f}")
    print(f"Precision  : {prec:.4f}")
    print(f"Recall     : {rec:.4f}")
    print(f"F1-score   : {f1:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"ROC-AUC    : {roc_auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Stayed", "Exited"]))

    return dict(y_proba=y_proba, y_pred=y_pred, cm=cm, fpr=fpr, tpr=tpr,
                roc_auc=roc_auc, accuracy=acc, precision=prec, recall=rec,
                f1=f1, specificity=specificity)


def plot_confusion_matrix(cm, title, out_path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm,
                            display_labels=["Stayed (0)", "Exited (1)"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_comparison(results_dict, out_path):
    plt.figure(figsize=(6.5, 6))
    colors = {"Baseline ANN": "#94a3b8", "Proposed ANN (SMOTE)": "#2563eb"}
    for label, r in results_dict.items():
        plt.plot(r["fpr"], r["tpr"], lw=2, color=colors.get(label),
                  label=f"{label} (AUC = {r['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Baseline vs Proposed (SMOTE)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_training_curves(history, title_prefix, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(history["loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_title(f"{title_prefix} — Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history["accuracy"], label="Train Accuracy")
    axes[1].plot(history["val_accuracy"], label="Val Accuracy")
    axes[1].set_title(f"{title_prefix} — Accuracy over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="models")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    data = np.load(os.path.join(args.models, "test_split.npz"))
    X_test, y_test = data["X_test"], data["y_test"]

    baseline_model = keras.models.load_model(os.path.join(args.models, "baseline_ann.keras"))
    proposed_model = keras.models.load_model(os.path.join(args.models, "proposed_ann_smote.keras"))

    with open(os.path.join(args.out, "baseline_history.json")) as f:
        baseline_hist = json.load(f)
    with open(os.path.join(args.out, "proposed_history.json")) as f:
        proposed_hist = json.load(f)

    baseline_results = evaluate_model(baseline_model, X_test, y_test, label="Baseline ANN")
    proposed_results = evaluate_model(proposed_model, X_test, y_test, label="Proposed ANN (SMOTE)")

    plot_confusion_matrix(baseline_results["cm"], "Confusion Matrix — Baseline ANN (Test Set)",
                           os.path.join(args.out, "confusion_matrix_baseline.png"))
    plot_confusion_matrix(proposed_results["cm"], "Confusion Matrix — Proposed ANN SMOTE (Test Set)",
                           os.path.join(args.out, "confusion_matrix_proposed.png"))

    plot_roc_comparison(
        {"Baseline ANN": baseline_results, "Proposed ANN (SMOTE)": proposed_results},
        os.path.join(args.out, "roc_curve_comparison.png")
    )

    plot_training_curves(baseline_hist, "Baseline ANN",
                          os.path.join(args.out, "training_curve_baseline.png"))
    plot_training_curves(proposed_hist, "Proposed ANN (SMOTE)",
                          os.path.join(args.out, "training_curve_proposed.png"))

    rows = []
    for label, r, hist in [("Baseline ANN (class_weight)", baseline_results, baseline_hist),
                            ("Proposed ANN (SMOTE)", proposed_results, proposed_hist)]:
        rows.append({
            "Model": label,
            "Accuracy": round(r["accuracy"], 4),
            "Precision": round(r["precision"], 4),
            "Recall": round(r["recall"], 4),
            "F1": round(r["f1"], 4),
            "Specificity": round(r["specificity"], 4),
            "ROC_AUC": round(r["roc_auc"], 4),
            "Epochs_trained": len(hist["loss"]),
        })
    results_df = pd.DataFrame(rows)
    results_df.to_csv(os.path.join(args.out, "results.csv"), index=False)
    print("\n=== Final Results Summary ===")
    print(results_df.to_string(index=False))
    print(f"\nSaved metrics table to {os.path.join(args.out, 'results.csv')}")
    print("Saved figures to", args.out)


if __name__ == "__main__":
    main()
