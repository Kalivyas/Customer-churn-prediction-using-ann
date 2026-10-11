# Bank Customer Churn Prediction using Artificial Neural Networks with SMOTE-Based Class Imbalance Correction

## 1. Project Title

**Bank Customer Churn Prediction using Artificial Neural Networks with SMOTE-Based Class Imbalance Correction**

## 2. Student Details

- Name: Kali Vyas
- Registration Number: 2430010353
- Section: D

## 3. Dataset

- **Dataset name:** Churn_Modelling.csv (Bank Customer Churn Prediction dataset)
- **Dataset source:** Kaggle — publicly available 10,000-row bank churn benchmark dataset
- **Number of samples:** 10,000
- **Number of classes:** 2 (binary classification — Exited: 1 = churned, 0 = retained)
- **Class distribution:** 7,963 stayed (79.63%) / 2,037 exited (20.37%) — imbalanced
- **Train / Validation / Test split:**
  - Test set: 2,000 samples (20%), stratified
  - Training pool: 8,000 samples (80%), of which 20% is further held out internally as
    validation during `model.fit()` (~6,400 train / ~1,600 val / 2,000 test)
- **Preprocessing performed:**
  1. Dropped identifier columns: RowNumber, CustomerId, Surname
  2. Median/mode imputation for missing values (dataset had none, but pipeline is defensive)
  3. Label-encoded Gender (binary)
  4. One-hot encoded Geography (Germany, Spain; France is reference category)
  5. Stratified 80/20 train/test split, `random_state=42`
  6. StandardScaler fit on training data only (no leakage)
  7. **[Proposed model only]** SMOTE oversampling applied to the scaled training set

Full details: `data/dataset_information.txt`

## 4. Model

- **Model/architecture name:** Feed-forward Artificial Neural Network (ANN / Multilayer Perceptron)
- **Architecture description:** Dense(16, ReLU) → Dense(8, ReLU) → Dense(1, Sigmoid), 11 input features
- **Pretrained model:** None — trained from scratch
- **Proposed modification (CO5 innovation):** A second model with the *identical* architecture and
  hyperparameters is trained on a SMOTE-oversampled training set instead of using `class_weight`,
  isolating the effect of data-level vs. loss-level imbalance correction. See Section 8.

## 5. Hyperparameters

| Hyperparameter   | Value |
| ---------------- | ----- |
| Learning rate     | 0.001 |
| Batch size        | 32 |
| Number of epochs  | 150 (capped), early stopping patience = 10 on val_loss |
| Optimizer         | Adam |
| Loss function     | Binary Crossentropy |
| Scheduler         | None (Adam's adaptive learning rate used as-is) |
| Dropout           | Not used in this architecture |
| Weight decay      | None by default; L1/L2 kernel regularization tested as a separate experiment (see `regularizer` argument in `src/model.py`) |
| Imbalance handling (Baseline) | Keras `class_weight` (inverse class frequency) |
| Imbalance handling (Proposed) | SMOTE oversampling of training set (8,000 → 12,740 samples, 50/50 balance) |
| Random seed        | 42 (NumPy and TensorFlow), fixed throughout |

## 6. Hyperparameter Tuning

- **Parameters considered:** learning rate, hidden layer sizes, regularization type (None / L1 / L2),
  imbalance-handling strategy (class_weight vs. SMOTE)
- **Values tested:** learning rate 0.001 (default, found stable); regularization λ = 0.001 for L1/L2
  comparison; hidden layers 16→8 chosen after informal testing against larger (32→16) and smaller
  (8→4) configurations, which showed either overfitting or underfitting on the validation split
- **Final selected values:** see hyperparameter table above
- **Reason for selection:** Adam + lr=0.001 gave the most stable convergence without loss spikes;
  the 16→8 architecture balanced capacity against the small 10k-row dataset size to avoid overfitting;
  SMOTE was selected over class_weight alone after both were compared head-to-head (Section 8) and
  SMOTE gave a better precision/F1/AUC trade-off

## 7. Results

Full metrics: `results/results.csv` | Confusion matrices: `results/confusion_matrix_baseline.png`,
`results/confusion_matrix_proposed.png` | ROC comparison: `results/roc_curve_comparison.png` |
Training curves: `results/training_curve_baseline.png`, `results/training_curve_proposed.png`

| Model | Accuracy | Precision | Recall | F1 | Specificity | ROC-AUC |
|---|---|---|---|---|---|---|
| Baseline ANN (class_weight) | 0.7745 | 0.4669 | 0.7617 | 0.5789 | 0.7778 | 0.8521 |
| **Proposed ANN (SMOTE)** | **0.8125** | **0.5303** | 0.6880 | **0.5989** | **0.8443** | **0.8540** |

### 5-Fold Stratified Cross-Validation (k = 5) - Baseline ANN

Run with `src/cross_validation.py` on the 8,000-sample training split (fresh model per fold, same
architecture, class_weight and early stopping). Files: `results/cv_results.csv`, `results/cv_results.png`.

| Fold | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| 1 | 0.7887 | 0.4877 | 0.7270 | 0.5837 | 0.8460 |
| 2 | 0.7938 | 0.4960 | 0.7607 | 0.6005 | 0.8663 |
| 3 | 0.8150 | 0.5323 | 0.7577 | 0.6253 | 0.8695 |
| 4 | 0.7906 | 0.4898 | 0.6656 | 0.5644 | 0.8327 |
| 5 | 0.7631 | 0.4481 | 0.7025 | 0.5472 | 0.8323 |
| **Mean +/- Std** | **0.7902 +/- 0.0165** | **0.4908 +/- 0.0268** | **0.7227 +/- 0.0356** | **0.5842 +/- 0.0273** | **0.8494 +/- 0.0160** |

The mean CV ROC-AUC (0.849) is consistent with the held-out test AUC (0.852), and the small standard
deviation across folds shows the baseline's performance is stable and not the product of a single
favourable train/test split.

## 8. Innovation 

**What is the innovation compared with the baseline/SOTA methods?**

The baseline ANN handles the dataset's 79.6/20.4 class imbalance only by reweighting the loss
function (`class_weight`), which does not change what data the network actually sees — it only
tells the optimizer to penalize minority-class mistakes more heavily.

The **proposed model** applies **SMOTE (Synthetic Minority Over-sampling Technique)** to the
*training set only* (after scaling, before fitting), generating synthetic "Exited" customer
examples in feature space so the network learns from a genuinely balanced 50/50 distribution
(8,000 → 12,740 training samples). The architecture, optimizer, learning rate, and epoch budget
are kept identical to the baseline, so any performance difference is attributable solely to this
change — this is what makes it a controlled, defensible ablation rather than an incidental
hyperparameter change.

**Ablation study (CO5 evidence):** see `results/ablation_study.csv`

| Experiment | Imbalance Handling | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|---|
| Baseline | class_weight only | 0.7745 | 0.4669 | 0.7617 | 0.5789 | 0.8521 |
| Proposed | SMOTE oversampling | 0.8125 | 0.5303 | 0.6880 | 0.5989 | 0.8540 |

**Interpretation:** SMOTE improves accuracy (+3.8 pts), precision (+6.3 pts), F1 (+2.0 pts), and
AUC (+0.19 pts), at the cost of recall (-7.4 pts). This is an explainable trade-off: class_weight
alone still trains on the same limited real minority examples and compensates by over-predicting
the minority class (inflating recall, hurting precision). SMOTE instead supplies new synthetic
minority examples for the network to learn actual decision-boundary structure from, yielding a
more balanced, higher-precision, higher-AUC classifier — arguably more useful for a bank that
wants confident targeting of at-risk customers rather than a high-recall, high-false-alarm model.

## 9. Comparison with State-of-the-Art (CO4)

Full table: `results/comparison.csv`. Summary:


| Model | Dataset | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|---|
| ANN, case study in banking [verify authors] | Same 10k Kaggle dataset | 0.860 | - | - | - | - |
| Random Forest (Tran, Le & Nguyen, 2023) [verify value] | Kaggle bank churn data | 0.9725 | - | - | - | - |
| Logistic Regression (Tran, Le & Nguyen, 2023) [verify value] | Kaggle bank churn data | 0.8727 | - | - | - | - |
| XGBoost (IJRASET, 2025) [confirm dataset] | Bank churn data [confirm] | 0.9985 | 0.9975 | 0.9950 | 0.9963 | 0.9974 |
| Voting Classifier, no SMOTE (Bhuria et al., 2025) | Same 10k Kaggle dataset | 0.87 | 0.87 | 0.80 | 0.87 | - |
| Voting Classifier + SMOTE (Bhuria et al., 2025) | Same 10k Kaggle dataset | 0.90 | 0.90 | 0.90 | 0.90 | - |
| Baseline ANN (Ours) | Churn_Modelling.csv | 0.7745 | 0.4669 | 0.7617 | 0.5789 | 0.8521 |
| **Proposed ANN + SMOTE (Ours)** | Churn_Modelling.csv | **0.8125** | 0.5303 | 0.6880 | 0.5989 | 0.8540 |

## 10. Folder Structure

```
RegistrationNumber/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── Churn_Modelling.csv
│   └── dataset_information.txt
│
├── src/
│   ├── train.py            # trains baseline (class_weight) and proposed (SMOTE) ANNs
│   ├── test.py              # evaluates both models, produces figures and results.csv
│   ├── cross_validation.py  # 5-fold stratified CV (k=5) for the baseline ANN
│   ├── model.py              # ANN architecture definition
│   └── preprocessing.py     # data loading, cleaning, encoding, scaling
│
├── notebooks/
│   └── experiments.ipynb    # end-to-end notebook mirroring src/train.py + src/test.py
│
├── results/
│   ├── results.csv                       # final metrics for both models
│   ├── comparison.csv                    # CO4 — comparison with SOTA papers
│   ├── ablation_study.csv                # CO5 — baseline vs SMOTE ablation
│   ├── cv_results.csv                    # 5-fold CV metrics per fold + mean/std
│   ├── cv_results.png                    # ROC-AUC per fold
│   ├── confusion_matrix_baseline.png
│   ├── confusion_matrix_proposed.png
│   ├── roc_curve_comparison.png
│   ├── training_curve_baseline.png
│   ├── training_curve_proposed.png
│   ├── baseline_history.json
│   └── proposed_history.json
│
├── models/
│   ├── baseline_ann.keras
│   ├── proposed_ann_smote.keras
│   ├── test_split.npz            # exact held-out test/train arrays for reproducibility
│   └── model_description.txt
│
└── figures/
    └── (copies of key result figures for quick viewing, see results/)
```

## How to Reproduce

```bash
pip install -r requirements.txt
python src/train.py --data data/Churn_Modelling.csv --out models
python src/test.py --models models --out results
python src/cross_validation.py --data data/Churn_Modelling.csv --out results
```

Note: the executed notebook's numbers can differ from the tables above by ~1-2 points (TensorFlow is not bit-for-bit deterministic across runs even with a fixed seed); the tables in this README come from the `src/` scripts.

Or open `notebooks/experiments.ipynb` and run all cells (self-contained: data cleaning, ANN layers, SMOTE and 5-fold CV are all written out inline).

## References 

1. Baby, B., Dawod, Z., Sharif, M. S., & Elmedani, W. (2023). *Customer churn prediction model using artificial neural network: a case study in banking.* 3ICT 2023, IEEE, pp. 154-161. 
2. Tran, H., Le, N., & Nguyen, V.-H. (2023). *Customer Churn Prediction in the Banking Sector Using Machine Learning-Based Classification Models.* Interdisciplinary Journal of Information, Knowledge, and Management (IJIKM), 18, 87-105. [VERIFY first-author initial and DOI on page 1]
3. *Machine Learning in Bank Customer Churn Prediction: Improving Accuracy through Feature Engineering.* International Journal for Research in Applied Science & Engineering Technology (IJRASET), Vol. 13, Issue II, February 2025.
4. Bhuria, R., Gupta, S., Kaur, U., Bharany, S., Rehman, A. U., Hussen, S., Tejani, G. G., & Jangir, P. (2025). *Ensemble-based customer churn prediction in banking: a voting classifier approach for improved client retention using demographic and behavioral data.* Discover Sustainability. Published online 14 January 2025. https://doi.org/10.1007/s43621-025-00807-8 
