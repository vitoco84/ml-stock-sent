from src.models.base import Base
from src.models.linreg import LinearElasticNet
from src.models.xgboost import XGBoost


MODELS = {
    "linreg": LinearElasticNet,
    "xgboost": XGBoost
}

def build_model(kind: str, **params) -> Base:
    k = kind.lower()
    if k not in MODELS:
        raise KeyError(f"Unknown model '{kind}'. Available: {list(MODELS)}")
    clean = {kk: vv for kk, vv in params.items() if vv is not None}
    return MODELS[k](**clean)
