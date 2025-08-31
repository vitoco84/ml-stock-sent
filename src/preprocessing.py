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
    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> IdentityTransformer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X

def get_preprocessor(X: pd.DataFrame, model_name: str) -> Tuple[Pipeline, list[str]]:
    """Build Preprocessor Pipeline."""

    # No-op  Transformer for preserving DatFrames and Column names
    if model_name.lower() in {"cnn", "lstm"}:
        return Pipeline([("identity", IdentityTransformer())], memory=None), list(X.columns)

    cat_features = [c for c in ["dow"] if c in X.columns]
    num_features = [c for c in X.columns if c not in cat_features + ["date"]]

    if not (num_features or cat_features):
        raise ValueError("No feature columns found after filtering (only targets/date present).")

    if model_name.lower() in {"random_forest", "xgboost"}:
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
