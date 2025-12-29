import numpy as np
import pandas as pd
import pytest

from mlfinance.features import rolling_indicators


def test_rolling_indicators_basic_shape_and_index():
    idx = pd.date_range("2025-01-01", periods=6, freq="D")
    x = pd.Series([1, 2, 3, 4, 5, 6], index=idx)

    out = rolling_indicators(x, window=3)

    assert isinstance(out, pd.DataFrame)
    assert out.index.equals(x.index)
    assert out.shape == (len(x), 4)  # mean, std, min, max


def test_rolling_indicators_nans_at_start():
    idx = pd.date_range("2025-01-01", periods=6, freq="D")
    x = pd.Series([1, 2, 3, 4, 5, 6], index=idx)

    out = rolling_indicators(x, window=3)

    # for window=3 with min_periods=window => first 2 rows are NaN for all stats
    first = out.iloc[:2]
    assert first.isna().all().all()


def test_rolling_indicators_invalid_window_raises():
    idx = pd.date_range("2025-01-01", periods=3, freq="D")
    x = pd.Series([1.0, 2.0, 3.0], index=idx)

    with pytest.raises(ValueError):
        rolling_indicators(x, window=0)

    with pytest.raises(ValueError):
        rolling_indicators(x, window=-1)

    with pytest.raises(ValueError):
        rolling_indicators(x, window=2.5)


def test_rolling_indicators_empty_input_returns_empty_df():
    x = pd.Series([], dtype=float)

    out = rolling_indicators(x, window=3)

    assert isinstance(out, pd.DataFrame)
    assert out.index.equals(x.index)
    assert out.shape == (0, 0)


def test_rolling_indicators_matches_public_reference():
    # "Pandas rolling computation
    idx = pd.date_range("2025-01-01", periods=7, freq="D")
    x = pd.Series([5, 7, 6, 8, 10, 9, 11], index=idx).astype(float)

    window = 3
    out = rolling_indicators(x, window=window)

    r = x.rolling(window=window, min_periods=window)
    expected = pd.DataFrame(
        {
            f"roll_mean_{window}": r.mean(),
            f"roll_std_{window}": r.std(ddof=0),
            f"roll_min_{window}": r.min(),
            f"roll_max_{window}": r.max(),
        },
        index=x.index,
    ).astype(float)

    assert out.shape == expected.shape
    assert out.index.equals(expected.index)

    assert np.allclose(
        out.to_numpy(),
        expected.to_numpy(),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
