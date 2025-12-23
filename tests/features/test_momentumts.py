import numpy as np
import pandas as pd
import pytest

import mlfinance.features as mf


def test_momentum_simple_computation():
    prices = pd.Series([100.0, 110.0, 121.0])

    out = mf.momentum_ts(prices, window=1)

    expected = pd.Series([np.nan, 0.10, 0.10], index=prices.index)

    pd.testing.assert_series_equal(out.iloc[:, 0], expected, check_names=False)


def test_momentum_initial_values_are_nan():
    prices = pd.Series([100, 101, 102, 103, 104], dtype=float)

    out = mf.momentum_ts(prices, window=3)

    assert out.iloc[:3, 0].isna().all()


def test_momentum_matches_pandas_pct_change():
    prices = pd.Series([100, 102, 101, 103, 105], dtype=float)

    out = mf.momentum_ts(prices, window=2).iloc[:, 0]
    expected = prices.pct_change(periods=2)

    pd.testing.assert_series_equal(out, expected, check_names=False)


def test_momentum_matches_simple_returns():
    prices = pd.Series([100, 102, 101, 103, 105], dtype=float)

    mom = mf.momentum_ts(prices, window=2).iloc[:, 0]
    ret = mf.returns(prices, horizons=[2], kind="simple").iloc[:, 0]

    pd.testing.assert_series_equal(mom, ret, check_names=False)


def test_momentum_invalid_window_raises():
    prices = pd.Series([100, 101, 102], dtype=float)

    with pytest.raises(ValueError):
        mf.momentum_ts(prices, window=0)

    with pytest.raises(TypeError):
        mf.momentum_ts(prices, window=1.5)
