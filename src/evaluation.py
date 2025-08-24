import shap
from sklearn.multioutput import MultiOutputRegressor


class SHAPExplainer:
    """Shapley Explainer."""

    def __init__(self, model, preprocessor, background_data, mode: str = "kernel", seed: int = 42):
        self.model = self._unwrap(model)
        self.preprocessor = preprocessor
        self.X_bg = self.preprocessor.transform(background_data)
        self.mode = mode
        self.seed = seed

    def explain(self, X):
        X_proc = self.preprocessor.transform(X) if self.preprocessor else X
        if isinstance(self.model, MultiOutputRegressor):
            return [self._explain_single(est, X_proc) for est in self.model.estimators_]
        else:
            return self._explain_single(self.model, X_proc)

    def _explain_single(self, model, X_proc):
        model = self._unwrap(model)

        if self.mode == "tree":
            explainer = shap.TreeExplainer(model)
        elif self.mode == "linear":
            explainer = shap.LinearExplainer(model, self.X_bg)
        elif self.mode == "deep":
            explainer = shap.DeepExplainer(model, self.X_bg[:50])
        else:
            explainer = shap.KernelExplainer(model.predict, self.X_bg[:50], seed=self.seed)

        return explainer.shap_values(X_proc)

    def _unwrap(self, model):
        return getattr(model, "model", model)
