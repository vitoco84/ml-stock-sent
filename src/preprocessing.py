from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class IdentityTransformer(BaseEstimator, TransformerMixin):
    """No-op transformer for models expecting raw DataFrames (e.g. CNN/LSTM)."""

    def __init__(self, copy: bool = False) -> None:
        self.copy = copy

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> IdentityTransformer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.copy() if self.copy else X

def get_preprocessor(X: pd.DataFrame, model_name: str) -> Tuple[Pipeline, list[str] | None]:
    """
    Build preprocessing pipeline for a given model type.

    - CNN/LSTM: no-op (preserve raw lagged features like log returns)
    - RandomForest/XGBoost: imputation only
    - Others (linear/MLP): imputation and scaling
    """

    # Torch sequence models expect raw lag_* features, no preprocessing
    if model_name.lower() in {"cnn", "lstm"}:
        return Pipeline([("identity", IdentityTransformer())], memory=None), list(X.columns)

    # Separate numeric vs categorical
    cat_features = [c for c in ["dow"] if c in X.columns]
    num_features = [c for c in X.columns if c not in cat_features + ["date"]]

    if not (num_features or cat_features):
        raise ValueError("No feature columns found after filtering (only targets/date present).")

    # Preprocessors
    if model_name.lower() in {"xgboost"}:
        num_tf = SimpleImputer(strategy="median")
    else:
        num_tf = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

    cat_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]) if cat_features else "passthrough"

    pre = ColumnTransformer(transformers=[
        ("num", num_tf, num_features),
        ("cat", cat_tf, cat_features),
    ])

    return Pipeline([("pre", pre)], memory=None), num_features + cat_features
