from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_preprocessor(X: pd.DataFrame, model_name: str) -> Tuple[Pipeline, list[str]]:
    """Build Preprocessor Pipeline."""
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
