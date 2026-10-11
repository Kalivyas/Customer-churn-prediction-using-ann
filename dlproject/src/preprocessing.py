"""
preprocessing.py
Loads Churn_Modelling.csv, cleans it, encodes categoricals, and returns
train/test splits ready for model training. Fit objects (scaler, encoders)
are NOT persisted separately here for simplicity — they are refit inside
train.py / test.py in an identical, reproducible way using RANDOM_STATE.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

RANDOM_STATE = 42
TARGET_COL = "Exited"
ID_COLS_TO_DROP = ["RowNumber", "CustomerId", "Surname"]


def load_raw(file_path: str) -> pd.DataFrame:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    elif ext == ".csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return df


def clean_and_encode(df: pd.DataFrame):
    df = df.copy()
    drop_cols = [c for c in ID_COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=drop_cols)

    candidate_cols = [c for c in df.columns if c != TARGET_COL]
    numeric_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in candidate_cols if not pd.api.types.is_numeric_dtype(df[c])]

    for c in numeric_cols:
        if df[c].isnull().sum() > 0:
            df[c] = df[c].fillna(df[c].median())
    for c in categorical_cols:
        if df[c].isnull().sum() > 0:
            df[c] = df[c].fillna(df[c].mode()[0])
    if df[TARGET_COL].isnull().sum() > 0:
        df = df.dropna(subset=[TARGET_COL])

    for c in categorical_cols:
        if df[c].nunique() <= 2:
            df[c] = LabelEncoder().fit_transform(df[c])
        else:
            dummies = pd.get_dummies(df[c], prefix=c, drop_first=True).astype(int)
            df = pd.concat([df.drop(columns=[c]), dummies], axis=1)

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    return df, feature_cols


def load_and_prepare(file_path: str, test_size: float = 0.2):
    """Returns X_train, X_test, y_train, y_test, feature_cols, scaler (fit on train only)."""
    df = load_raw(file_path)
    df, feature_cols = clean_and_encode(df)

    X = df[feature_cols].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, feature_cols, scaler


def get_class_weight(y):
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    return {0: len(y) / (2 * n_neg), 1: len(y) / (2 * n_pos)}
