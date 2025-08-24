import shap
from sklearn.multioutput import MultiOutputRegressor


class SHAPExplainer:
    """Shapley Explainer."""

    def __init__(self, model, preprocessor, background_data, mode: str = "kernel", seed: int = 42):
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
            explainer = shap.KernelExplainer(model.predict, self.X_bg[:50], seed=self.seed)
        return explainer.shap_values(X_proc)

    def _unwrap(self, model):
        return getattr(model, "model", model)
