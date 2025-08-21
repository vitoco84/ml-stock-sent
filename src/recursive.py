from typing import Any, List, Optional

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from src.features import generate_full_feature_row
from src.sentiment import FinBERT


class RecursiveForecaster:
    """Recursive forecasting using a single-step model."""

    def __init__(
            self,
            model: Any,
            preprocessor: Any,
            y_scaler: Optional[Any] = None,
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
        df = df.copy()
        if not self._expected_cols:
            return df
        for c in self._expected_cols:
            if c not in df.columns:
                df[c] = 0.0
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
            feat_row = generate_full_feature_row(
                price_df=pdf,
                news_df=ndf,
                sentiment_model=self.sentiment_model,
                horizon=1
            )
            feat_row = self._align_feature_row(feat_row)

            X = self.preprocessor.transform(feat_row)
            y_pred = self.model.predict(X)
            y_pred = np.asarray(y_pred).reshape(-1, 1)

            if self.y_scaler is not None:
                y_pred = self.y_scaler.inverse_transform(y_pred)

            lr = float(y_pred.ravel()[0])
            lr_path.append(lr)

            next_date = (last_date + BDay(1)).normalize()
            next_price = last_price * np.exp(lr)

            pdf = pd.concat([pdf, pd.DataFrame([{
                "date": next_date,
                "adj_close": next_price,
                "open": last_price,
                "high": max(last_price, next_price),
                "low": min(last_price, next_price),
                "close": next_price,
                "volume": pdf["volume"].iloc[-1]
            }])], ignore_index=True)

            last_price, last_date = next_price, next_date

        return np.array(lr_path, dtype=float)
