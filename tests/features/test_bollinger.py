import numpy as np
import pandas as pd
import pytest

import mlfinance.features as mf
import mlfinance.math_utils as mu


def test_bollinger_preserves_index_and_length():
    x = pd.Series([1, 2, 3, 4, 5], index=pd.date_range("2024-01-01", periods=5))
    out = mf.bollinger_bands(x, window=3, n_std=2.0)

    assert isinstance(out, pd.DataFrame)
    assert out.index.equals(x.index)
    assert len(out) == len(x)


def test_bollinger_matches_definition():
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    window = 3
    n_std = 2.0

    out = mf.bollinger_bands(x, window=window, n_std=n_std)

    mid = mu.sma(x, window=window)
    sd = mu.rolling_std(x, window=window, ddof=0)

    expected_upper = mid + n_std * sd
    expected_lower = mid - n_std * sd

    pd.testing.assert_series_equal(out["bb_mid"], mid, check_names=False)
    pd.testing.assert_series_equal(out["bb_upper"], expected_upper, check_names=False)
    pd.testing.assert_series_equal(out["bb_lower"], expected_lower, check_names=False)


def test_bollinger_constant_series():
    x = pd.Series([5.0] * 10)
    out = mf.bollinger_bands(x, window=4, n_std=2.0)

    valid = out.dropna()
    assert (valid["bb_mid"] == 5.0).all()
    assert (valid["bb_upper"] == 5.0).all()
    assert (valid["bb_lower"] == 5.0).all()


def test_bollinger_invalid_window_raises():
    x = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        mf.bollinger_bands(x, window=0, n_std=2.0)


def test_bollinger_matches_ta_package():
    pytest.importorskip("ta")
    from ta.volatility import BollingerBands

    close = pd.Series([1.0, 2.0, 3.0, 2.0, 5.0, 6.0, 4.0, 3.0, 2.0, 1.0])
    window = 3
    n_std = 2.0

    ours = mf.bollinger_bands(close, window=window, n_std=n_std)

    ind = BollingerBands(close=close, window=window, window_dev=n_std)
    ref_mid = ind.bollinger_mavg()
    ref_upper = ind.bollinger_hband()
    ref_lower = ind.bollinger_lband()

    # Use allclose to be robust to tiny float differences; keep NaNs aligned
    assert np.allclose(ours["bb_mid"].to_numpy(), ref_mid.to_numpy(), equal_nan=True)
    assert np.allclose(ours["bb_upper"].to_numpy(), ref_upper.to_numpy(), equal_nan=True)
    assert np.allclose(ours["bb_lower"].to_numpy(), ref_lower.to_numpy(), equal_nan=True)
