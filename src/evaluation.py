import pandas as pd
import shap


def run_shap(model, preprocessor, X_raw, sample_size=50, background_size=20, y_scaler=None):
    """Evaluate SHAP."""
    # Preprocess features
    X_preprocessed = preprocessor.transform(X_raw)
    feature_names = getattr(preprocessor, "get_feature_names_out", lambda: X_raw.columns)()

    # Wrap into DataFrame so SHAP knows feature names
    X_sample = pd.DataFrame(X_preprocessed[:sample_size], columns=feature_names)

    # Background for KernelExplainer
    # background = shap.kmeans(X_sample, background_size)
    background = shap.kmeans(X_sample, min(background_size, len(X_sample)))

    # Explainer
    explainer = shap.KernelExplainer(model.predict, background)
    shap_values = explainer(X_sample)

    # If y_scaler is provided and this is single-output → rescale SHAP values
    if y_scaler is not None and shap_values.values.ndim == 2:
        shap_values.values = shap_values.values * y_scaler.scale_
        shap_values.base_values = shap_values.base_values * y_scaler.scale_ + y_scaler.mean_

    return shap_values
