import numpy as np
import pandas as pd


def calculate_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_return"] = np.log(df["adj_close"] / df["adj_close"].shift(1))
    return df
