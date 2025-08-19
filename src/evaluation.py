import shap
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor

class SHAPExplainer:
    def __init__(self, model, preprocessor, background_data):
        self.model = self._unwrap(model)
        self.preprocessor = preprocessor
        self.X_bg = self.preprocessor.transform(background_data)

    def explain(self, X):
        X_proc = self.preprocessor.transform(X)

        if isinstance(self.model, MultiOutputRegressor):
            return [self._explain_single(est, X_proc) for est in self.model.estimators_]
        else:
            return self._explain_single(self.model, X_proc)

    def _explain_single(self, model, X_proc):
        model = self._unwrap(model)
        model_type = self._infer_model_type(model)

        if model_type == "tree":
            explainer = shap.Explainer(model, self.X_bg)
        elif model_type == "linear":
            explainer = shap.LinearExplainer(model, self.X_bg)
        else:
            explainer = shap.KernelExplainer(model.predict, self.X_bg[:50])

        return explainer.shap_values(X_proc)

    def _unwrap(self, model):
        return getattr(model, "model", model)

    def _infer_model_type(self, model):
        try:
            from xgboost import XGBRegressor
            from lightgbm import LGBMRegressor
        except ImportError:
            XGBRegressor = LGBMRegressor = object

        if isinstance(model, (RandomForestRegressor, XGBRegressor, LGBMRegressor)):
            return "tree"
        elif isinstance(model, (ElasticNet, LinearRegression)):
            return "linear"
        else:
            return "kernel"
