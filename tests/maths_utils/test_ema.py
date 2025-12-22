import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from mlfinance.math_utils import ema


# Input validation
def test_ema_raises_if_prices_not_series():
    with pytest.raises(TypeError):
        ema(prices=[1, 2, 3], span=3)


def test_ema_raises_if_prices_not_numeric():
    prices = pd.Series(["a", "b", "c"])
    with pytest.raises(ValueError):
        ema(prices, span=3)


def test_ema_raises_if_span_invalid():
    prices = pd.Series([1, 2, 3])
    with pytest.raises(ValueError):
        ema(prices, span=0)


def test_ema_raises_if_min_periods_invalid():
    prices = pd.Series([1, 2, 3])
    with pytest.raises(ValueError):
        ema(prices, span=3, min_periods=-1)


# Output validation
def test_ema_output_length_index_dtype():
    prices = pd.Series(
        np.linspace(100, 110, 50), index=pd.date_range("2020-01-01", periods=50), dtype=float
    )
    span = 10
    ema_series = ema(prices, span=span)
    assert len(ema_series) == len(prices)
    assert ema_series.index.equals(prices.index)
    assert ema_series.dtype == float


def test_ema_matches_pandas_ewm_adjust_false():
    prices = pd.Series([1, 2, 3, 2, 5, 4, 6], dtype=float)
    span = 3
    min_periods = span

    got = ema(prices, span=span, min_periods=min_periods)
    exp = prices.ewm(span=span, adjust=False, min_periods=min_periods).mean()

    assert_allclose(got.to_numpy(), exp.to_numpy(), equal_nan=True)


def test_ema_constant_series_after_warmup():
    prices = pd.Series([100.0] * 50)
    span = 5
    mp = span

    got = ema(prices, span=span, min_periods=mp)
    exp = prices.ewm(span=span, adjust=False, min_periods=mp).mean()

    assert_allclose(got.to_numpy(), exp.to_numpy(), equal_nan=True)


def test_ema_nan_handling_matches_pandas_ignore_na_true_with_mask():
    prices = pd.Series([1, 2, np.nan, 4, 5, np.nan, np.nan, 8], dtype=float)
    span = 3

    got = ema(prices, span=span)

    exp = prices.ewm(span=span, adjust=False, min_periods=0, ignore_na=True).mean()
    exp = exp.mask(prices.isna())

    assert_allclose(got.to_numpy(), exp.to_numpy(), equal_nan=True)
