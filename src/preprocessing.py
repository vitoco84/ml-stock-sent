import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_preprocessor(X: pd.DataFrame) -> Pipeline:
    """Build a preprocessing pipeline using ColumnTransformer."""
    cat_features = [c for c in ["dow", "quarter"] if c in X.columns]
    num_features = [c for c in X.columns if c not in cat_features and c not in ["date", "target"]]

    num_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]) if cat_features else "passthrough"

    pre = ColumnTransformer(transformers=[
        ("num", num_tf, num_features),
        ("cat", cat_tf, cat_features),
    ])

    return Pipeline([("pre", pre)])
