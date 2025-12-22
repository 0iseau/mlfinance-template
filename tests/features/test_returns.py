import numpy as np
import pandas as pd
import pytest

import mlfinance.features as mf


def test_returns_preserves_index_and_length():
    x = pd.Series([100.0, 101.0, 103.0, 102.0], index=pd.date_range("2024-01-01", periods=4))
    out = mf.returns(x, horizons=(1, 2), kind="both")

    assert isinstance(out, pd.DataFrame)
    assert out.index.equals(x.index)
    assert len(out) == len(x)


def test_returns_columns_are_created():
    x = pd.Series([100.0, 101.0, 103.0, 102.0])
    out = mf.returns(x, horizons=(1, 2), kind="both")

    assert "ret_1" in out.columns
    assert "ret_2" in out.columns
    assert "logret_1" in out.columns
    assert "logret_2" in out.columns


def test_returns_values_simple_case_h1():
    x = pd.Series([100.0, 110.0, 121.0])
    out = mf.returns(x, horizons=(1,), kind="both")

    # simple return
    expected_ret = pd.Series([np.nan, 0.10, 0.10], index=x.index)
    pd.testing.assert_series_equal(out["ret_1"], expected_ret, check_names=False)

    # log return
    expected_log = pd.Series([np.nan, np.log(110.0 / 100.0), np.log(121.0 / 110.0)], index=x.index)
    pd.testing.assert_series_equal(out["logret_1"], expected_log, check_names=False)


def test_returns_constant_prices_are_zero_after_warmup():
    x = pd.Series([5.0] * 10)
    out = mf.returns(x, horizons=(1, 5), kind="both")

    for col in ["ret_1", "ret_5", "logret_1", "logret_5"]:
        valid = out[col].dropna()
        assert np.allclose(valid.to_numpy(), 0.0)


def test_returns_invalid_horizon_raises():
    x = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        mf.returns(x, horizons=(0,), kind="both")


def test_returns_matches_pandas_pct_change_simple():
    prices = pd.Series([100.0, 110.0, 121.0, 132.0])
    out = mf.returns(prices, horizons=(1,), kind="simple")
    expected = prices.pct_change()
    pd.testing.assert_series_equal(out["ret_1"], expected, check_names=False)
