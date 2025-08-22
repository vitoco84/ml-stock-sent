from src.models.base import Base
from src.models.direct_multi import DirectMultiStep
from src.models.linreg import LinearElasticNet
from src.models.xgboost import XGBoost


MODELS = {
    "linreg": LinearElasticNet,
    "xgboost": XGBoost,
}

DIRECT_MULTI_MODELS = {"xgboost", "rf"}

def build_model(kind: str, **params) -> Base:
    k = kind.lower()
    if k not in MODELS:
        raise KeyError(f"Unknown model '{kind}'. Available: {list(MODELS)}")

    base_cls = MODELS[k]
    horizon = params.get("horizon", 30)

    if k in DIRECT_MULTI_MODELS:
        base_params = {kk: vv for kk, vv in params.items() if kk not in {"horizon"}}
        return DirectMultiStep(
            base_cls,
            base_params=base_params,
            horizon=horizon,
            random_state=params.get("random_state", 42)
        )

    return base_cls(**params)
