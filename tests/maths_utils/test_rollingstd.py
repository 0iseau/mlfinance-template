import numpy as np
import pandas as pd

import mlfinance.math_utils as mu


def test_sma_preserves_index_and_length():
    x = pd.Series([1, 2, 3, 4, 5], index=pd.date_range("2024-01-01", periods=5))
    out = mu.sma(x, window=3)

    assert isinstance(out, pd.Series)
    assert out.index.equals(x.index)
    assert len(out) == len(x)


def test_sma_values_simple_case():
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = mu.sma(x, window=3)

    expected = pd.Series(
        [np.nan, np.nan, 2.0, 3.0, 4.0],
        index=x.index,
    )

    pd.testing.assert_series_equal(out, expected)


def test_sma_constant_series():
    x = pd.Series([5.0] * 10)
    out = mu.sma(x, window=4)

    # After warm-up, SMA of a constant series equals the constant
    valid = out.dropna()
    assert (valid == 5.0).all()
