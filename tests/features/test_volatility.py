import numpy as np
import pandas as pd
import pytest

import mlfinance.features as ft


def test_volatility_constant_prices_is_zero():
    prices = pd.Series([100.0] * 10)
    vol = ft.volatility(prices, kind="log")

    assert np.isfinite(vol)
    assert vol == 0.0


def test_volatility_matches_std_of_returns_log():
    prices = pd.Series([100.0, 110.0, 121.0, 133.1])
    vol = ft.volatility(prices, kind="log")

    logret = (np.log(prices) - np.log(prices.shift(1))).dropna()
    expected = float(logret.std(ddof=0))

    assert np.isclose(vol, expected)


def test_volatility_invalid_kind_raises():
    prices = pd.Series([100.0, 101.0, 102.0])
    with pytest.raises(ValueError):
        ft.volatility(prices, kind="bad")


def test_rolling_volatility_preserves_index_and_length():
    prices = pd.Series(
        [100.0, 101.0, 102.0, 103.0, 104.0],
        index=pd.date_range("2024-01-01", periods=5),
    )
    out = ft.rolling_volatility(prices, window=3, kind="log")

    assert isinstance(out, pd.Series)
    assert out.index.equals(prices.index)
    assert len(out) == len(prices)


def test_rolling_volatility_leading_nans_until_window():
    prices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
    out = ft.rolling_volatility(prices, window=3, kind="log")

    # log returns introduce 1 NaN, rolling std needs `window` points
    assert out.iloc[:3].isna().all()
    assert out.iloc[3:].notna().any()


def test_rolling_volatility_constant_prices_is_zero_after_warmup():
    prices = pd.Series([100.0] * 20)
    out = ft.rolling_volatility(prices, window=5, kind="log")

    valid = out.dropna()
    assert np.allclose(valid.to_numpy(), 0.0)


def test_rolling_volatility_matches_pandas_reference():
    prices = pd.Series([100.0, 110.0, 121.0, 133.1, 146.41, 161.051])
    window = 3

    out = ft.rolling_volatility(prices, window=window, kind="log")

    # PUBLIC reference: pandas rolling std on log returns
    logret = np.log(prices) - np.log(prices.shift(1))
    expected = logret.rolling(window=window, min_periods=window).std(ddof=0)

    pd.testing.assert_series_equal(out, expected, check_names=False)


def test_rolling_volatility_invalid_window_raises():
    prices = pd.Series([100.0, 101.0, 102.0])
    with pytest.raises(ValueError):
        ft.rolling_volatility(prices, window=0, kind="log")
