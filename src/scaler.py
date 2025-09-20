from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class SafeStandardScaler(StandardScaler):
    """
    Drop-in replacement for sklearn's StandardScaler that:
      - Accepts 1D arrays by reshaping to (n, 1)
      - Flattens back to 1D on inverse_transform if output has one column
    """

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None, sample_weight: Optional[np.ndarray] = None):
        return super().fit(self._ensure_2d(X), y, sample_weight=sample_weight)

    def transform(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        return super().transform(self._ensure_2d(X), **kwargs)

    def inverse_transform(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        out = super().inverse_transform(self._ensure_2d(X), **kwargs)
        return out.ravel() if out.shape[1] == 1 else out

    def fit_transform(self, X: pd.DataFrame, y: Optional[np.ndarray] = None, **fit_params) -> np.ndarray:
        return super().fit_transform(self._ensure_2d(X), y, **fit_params)

    @staticmethod
    def _ensure_2d(X) -> np.ndarray:
        """Ensure input is 2D (reshape 1D arrays to (n, 1))."""
        X = np.asarray(X)
        return X.reshape(-1, 1) if X.ndim == 1 else X
