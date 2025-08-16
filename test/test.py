import pandas as pd

from src.data import _rename_columns, load_prices_sentiment
from src.utils import is_cuda_available


def test_is_cuda_available():
    assert is_cuda_available() == True

def test_rename_columns():
    data = [
        {"Name": "Alice", "Price": 25, "Adj Price": 32564, "Description Is": ""}
    ]
    df = pd.DataFrame(data)
    _rename_columns(df)
    assert df.columns.tolist() == ["name", "price", "adj_price", "description_is"]

def test_exception_and_logger():
    df = load_prices_sentiment("foobar")
    assert df.empty == True
