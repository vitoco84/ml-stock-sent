import pandas as pd
import shap
from sklearn.multioutput import MultiOutputRegressor
from statsmodels.tsa.stattools import adfuller


class SHAPExplainer:
    """Shapley Explainer."""

    def __init__(self, model, preprocessor, background_data, mode: str = "tree", seed: int = 42):
        self.model = model
        self.preprocessor = preprocessor
        self.mode = mode
        self.seed = seed

        if self.preprocessor is not None:
            self.X_bg = self.preprocessor.transform(background_data)
        else:
            self.X_bg = background_data

    def explain(self, X):
        X_proc = self.preprocessor.transform(X) if self.preprocessor else X
        model = self._unwrap(self.model)
        if isinstance(model, MultiOutputRegressor):
            return [self._explain_single(est, X_proc) for est in model.estimators_]
        return self._explain_single(model, X_proc)

    def _explain_single(self, model, X_proc):
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
    def _unwrap(model):
        return getattr(model, "model", model)

def adf_test(series: pd.Series, name: str = "series", as_dict: bool = False):
    """Augmented Dickey-Fuller."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    stat, pval, lags, nobs, crit, _ = adfuller(s, autolag="AIC", regression="c")

    if as_dict:
        return {
            "name": name,
            "adf_stat": stat,
            "p_value": pval,
            "lags_used": lags,
            "n_obs": nobs,
            "crit_values": crit,
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
