import numpy as np
import pandas as pd
import pytest

from mlfinance.features import lags


def test_lags_preserves_index_and_shape():
    idx = pd.date_range("2025-01-01", periods=5, freq="D")
    x = pd.Series([10, 20, 30, 40, 50], index=idx)

    out = lags(x, shift=2)

    assert isinstance(out, pd.DataFrame)
    assert out.index.equals(x.index)
    assert out.shape == (5, 1)


def test_lags_shift_values_are_correct():
    idx = pd.date_range("2025-01-01", periods=5, freq="D")
    x = pd.Series([10, 20, 30, 40, 50], index=idx)

    out = lags(x, shift=2)

    expected = x.shift(2)
    result = out.iloc[:, 0]

    assert np.allclose(
        result.to_numpy(),
        expected.to_numpy(),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )


def test_lags_empty_input_returns_empty_df():
    x = pd.Series([], dtype=float)

    out = lags(x, shift=2)

    assert isinstance(out, pd.DataFrame)
    assert out.index.equals(x.index)
    assert out.shape == (0, 0)


def test_lags_invalid_shift_raises():
    idx = pd.date_range("2025-01-01", periods=3, freq="D")
    x = pd.Series([1.0, 2.0, 3.0], index=idx)

    with pytest.raises(ValueError):
        lags(x, shift=-1)

    with pytest.raises(ValueError):
        lags(x, shift=1.5)  # type: ignore[arg-type]


def test_lags_shift_zero_means_full_length_shift():
    idx = pd.date_range("2025-01-01", periods=4, freq="D")
    x = pd.Series([1, 2, 3, 4], index=idx)

    out = lags(x, shift=0)

    # per your implementation: shift=0 => shift=data.size
    expected = x.shift(x.size)
    result = out.iloc[:, 0]

    assert np.allclose(
        result.to_numpy(),
        expected.to_numpy(),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )


def test_lags_matches_public_reference_pandas_shift():
    # "Public classic function" reference: pandas.Series.shift
    idx = pd.date_range("2025-01-01", periods=6, freq="D")
    x = pd.Series([5, 7, 6, 8, 10, 9], index=idx)

    out = lags(x, shift=3)
    result = out.iloc[:, 0]

    expected = x.shift(3)

    assert np.allclose(
        result.to_numpy(),
        expected.to_numpy(),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
