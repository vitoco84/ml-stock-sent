from typing import List, Optional

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from src.features import generate_full_feature_row
from src.sentiment import FinBERT


class RecursiveForecaster:
    """Recursive forecasting using a single-step model."""

    def __init__(
            self,
            model: object,
            preprocessor: object,
            y_scaler: Optional[object] = None,
            sentiment_model: Optional[FinBERT] = None,
    ) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.y_scaler = y_scaler
        self.sentiment_model = sentiment_model

        self._expected_cols: List[str] = list(
            getattr(preprocessor, "feature_names_in_", [])
        )

    def _align_feature_row(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add any missing expected columns (fill 0.0), drop extras, then reorder."""
        if not self._expected_cols:
            return df

        missing = [c for c in self._expected_cols if c not in df.columns]
        if missing:
            for c in missing:
                df[c] = 0.0

        extra = [c for c in df.columns if c not in self._expected_cols]
        if extra:
            df = df.drop(columns=extra)

        return df[self._expected_cols]

    def forecast(
            self,
            price_df: pd.DataFrame,
            news_df: Optional[pd.DataFrame],
            horizon: int,
    ) -> np.ndarray:
        pdf = price_df.copy()
        ndf = news_df.copy() if news_df is not None else pd.DataFrame(columns=["date", "headline"])
        pdf["date"] = pd.to_datetime(pdf["date"]).dt.normalize()
        if not ndf.empty:
            ndf["date"] = pd.to_datetime(ndf["date"]).dt.normalize()

        if pdf.empty or "adj_close" not in pdf.columns:
            raise ValueError("price_df must contain at least one row with 'adj_close'.")

        lr_path: list[float] = []
        last_price = float(pdf["adj_close"].iloc[-1])
        last_date = pd.to_datetime(pdf["date"].iloc[-1])

        for _ in range(horizon):
            feat_row = generate_full_feature_row(pdf, ndf, self.sentiment_model, horizon=1)
            feat_row = self._align_feature_row(feat_row)

            X = self.preprocessor.transform(feat_row)
            y_pred = self.model.predict(X)

            if self.y_scaler is not None:
                y_pred = self.y_scaler.inverse_transform(y_pred)

            y_pred = np.asarray(y_pred).ravel()
            lr = float(y_pred[0])
            lr_path.append(lr)

            next_date = (last_date + BDay(1)).normalize()
            next_price = last_price * np.exp(lr)

            pdf = pd.concat([pdf, pd.DataFrame([{
                "date": next_date,
                "adj_close": next_price,
                "open": np.nan,
                "high": np.nan,
                "low": np.nan,
                "close": np.nan,
                "volume": np.nan
            }])], ignore_index=True)

            last_price, last_date = next_price, next_date

        return np.array(lr_path, dtype=float)
