import numpy as np
import pandas as pd
import pytest

from mlfinance.math_utils import rolling_std


def test_rolling_std_basic_shape_and_index():
    idx = pd.date_range("2025-01-01", periods=6, freq="D")
    x = pd.Series([1, 2, 3, 4, 5, 6], index=idx, dtype=float)

    out = rolling_std(x, 3, df=1)

    assert isinstance(out, pd.Series)
    assert out.index.equals(x.index)
    assert out.shape == x.shape
    assert out.dtype == float


def test_rolling_std_invalid_inputs_raise():
    x = pd.Series([1.0, 2.0, 3.0])

    with pytest.raises(TypeError):
        rolling_std([1, 2, 3], 3)  # type: ignore

    with pytest.raises(ValueError):
        rolling_std(x, 0)

    with pytest.raises(ValueError):
        rolling_std(x, 3, df=2)  # type: ignore

    with pytest.raises(ValueError):
        rolling_std(x, 3, df=1.5)  # type: ignore


def test_rolling_std_constant_series_is_nan_then_zero():
    idx = pd.date_range("2025-01-01", periods=5, freq="D")
    x = pd.Series([10, 10, 10, 10, 10], index=idx, dtype=float)

    out = rolling_std(x, 3, df=0)

    # First two rows are NaN due to min_periods=window
    assert out.iloc[:2].isna().all()

    # After enough periods, std of constant series = 0
    assert np.allclose(out.iloc[2:].to_numpy(), 0.0, rtol=0.0, atol=0.0, equal_nan=False)


def test_rolling_std_matches_public_reference():
    # Public reference: pandas rolling std
    idx = pd.date_range("2025-01-01", periods=7, freq="D")
    x = pd.Series([4, 7, 6, 8, 10, 9, 11], index=idx, dtype=float)

    w = 3
    out = rolling_std(x, w, df=0)

    expected = x.rolling(window=w, min_periods=w).std(ddof=0)

    assert np.allclose(
        out.to_numpy(),
        expected.to_numpy(),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
