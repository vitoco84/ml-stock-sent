import numpy as np
from sklearn.preprocessing import StandardScaler


class SafeStandardScaler(StandardScaler):
    """
    A drop-in replacement for StandardScaler that safely reshapes 1D inputs
    to 2D and flattens 2D output back to 1D if needed.
    """

    def fit(self, X, y=None, sample_weight=None):
        return super().fit(self._ensure_2d(X), y, sample_weight)

    def transform(self, X, copy=None):
        return super().transform(self._ensure_2d(X), copy)

    def inverse_transform(self, X, copy=None):
        result = super().inverse_transform(self._ensure_2d(X), copy)
        return result.ravel() if result.shape[1] == 1 else result

    def fit_transform(self, X, y=None, **fit_params):
        return super().fit_transform(self._ensure_2d(X), y, **fit_params)

    def _ensure_2d(self, X):
        X = np.asarray(X)
        return X.reshape(-1, 1) if X.ndim == 1 else X
