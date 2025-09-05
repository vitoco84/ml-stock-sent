from typing import Any

import pandas as pd
import shap
from sklearn.base import TransformerMixin
from sklearn.multioutput import MultiOutputRegressor
from statsmodels.tsa.stattools import adfuller


class SHAPExplainer:
    """Wrapper around SHAP explainers supporting multi-output models."""

    def __init__(
            self,
            model: Any,
            preprocessor: TransformerMixin | None,
            background_data: pd.DataFrame,
            mode: str = "tree",
            seed: int = 42,
            background_sample_size: int = 50
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.mode = mode
        self.seed = seed
        self.background_sample_size = background_sample_size

        if self.preprocessor is not None:
            self.X_bg = self.preprocessor.transform(background_data)
        else:
            self.X_bg = background_data

        # Optionally subsample background to avoid memory blow-up
        if hasattr(self.X_bg, "__len__") and len(self.X_bg) > background_sample_size:
            self.X_bg = self.X_bg[:background_sample_size]

    def explain(self, X: pd.DataFrame) -> list:
        """Return SHAP values for X. Always returns a list (one per output)."""
        X_proc = self.preprocessor.transform(X) if self.preprocessor else X
        model = self._unwrap(self.model)

        if isinstance(model, MultiOutputRegressor):
            return [self._explain_single(est, X_proc) for est in model.estimators_]
        return [self._explain_single(model, X_proc)]

    def _explain_single(self, model: Any, X_proc) -> Any:
        model = self._unwrap(model)
        if self.mode == "tree":
            explainer = shap.TreeExplainer(model)
        elif self.mode == "linear":
            explainer = shap.LinearExplainer(model, self.X_bg)
        elif self.mode == "deep":
            explainer = shap.DeepExplainer(model, self.X_bg[:50])
        else:
            explainer = shap.KernelExplainer(model.predict, self.X_bg[:50])
        return explainer.shap_values(X_proc)

    @staticmethod
    def _unwrap(model: Any) -> Any:
        """Unwrap nested model objects (e.g., sklearn wrappers)."""
        return getattr(model, "model", model)

def adf_test(series: pd.Series, name: str = "series", as_dict: bool = False):
    """Augmented Dickey-Fuller test for stationarity."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 10:
        raise ValueError(f"Series '{name}' too short for ADF test.")

    stat, pval, lags, nobs, crit, _ = adfuller(s, autolag="AIC", regression="c")

    if as_dict:
        return {
            "name": name,
            "adf_stat": float(stat),
            "p_value": float(pval),
            "lags_used": int(lags),
            "n_obs": int(nobs),
            "crit_values": {str(k): float(v) for k, v in crit.items()}
        }

    out = (
        f"ADF Test on '{name}'\n"
        f"{'-' * 40}\n"
        f"Test Statistic : {stat:.4f}\n"
        f"p-value        : {pval:.4g}\n"
        f"Lags Used      : {lags}\n"
        f"Observations   : {nobs}\n"
        f"{'-' * 40}\n"
    )
    for k, v in crit.items():
        out += f"Critical Value {k} : {v:.4f}\n"
    return out
