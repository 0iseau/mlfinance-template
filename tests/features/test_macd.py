import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from mlfinance.features import macd
from mlfinance.math_utils import ema


def test_macd_raises_on_invalid_params():
    prices = pd.Series([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        macd(prices, fast=0)
    with pytest.raises(ValueError):
        macd(prices, slow=0)
    with pytest.raises(ValueError):
        macd(prices, signal=0)
    with pytest.raises(ValueError):
        macd(prices, fast=26, slow=26)


def test_macd_preserves_index_length_and_dtype():
    idx = pd.date_range("2020-01-01", periods=120, freq="D")
    prices = pd.Series(np.linspace(100, 130, len(idx)), index=idx, dtype=float)

    macd_line, signal_line, hist = macd(prices)

    for s in (macd_line, signal_line, hist):
        assert s.index.equals(prices.index)
        assert len(s) == len(prices)
        assert s.dtype == float


def test_macd_hist_is_macd_minus_signal_on_valid_points():
    prices = pd.Series(np.linspace(100, 200, 300), dtype=float)

    macd_line, signal_line, hist = macd(prices)

    mask = ~(macd_line.isna() | signal_line.isna() | hist.isna())
    assert mask.any()
    assert (hist[mask] == (macd_line - signal_line)[mask]).all()


def test_macd_matches_reference_built_from_ema():
    prices = pd.Series(np.linspace(100, 200, 400), dtype=float)
    fast, slow, signal = 12, 26, 9

    macd_line, signal_line, hist = macd(prices, fast=fast, slow=slow, signal=signal)

    ema_fast = ema(prices, span=fast, min_periods=fast)
    ema_slow = ema(prices, span=slow, min_periods=slow)
    macd_ref = ema_fast - ema_slow

    signal_ref = ema(macd_ref, span=signal, min_periods=fast + signal)
    hist_ref = macd_ref - signal_ref

    assert_allclose(macd_line.to_numpy(), macd_ref.to_numpy(), equal_nan=True)
    assert_allclose(signal_line.to_numpy(), signal_ref.to_numpy(), equal_nan=True)
    assert_allclose(hist.to_numpy(), hist_ref.to_numpy(), equal_nan=True)


def test_macd_preserves_nan_positions_from_prices():
    prices = pd.Series([1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10] * 30, dtype=float)

    macd_line, signal_line, hist = macd(prices)

    nan_pos = prices.isna()
    assert macd_line[nan_pos].isna().all()
    assert signal_line[nan_pos].isna().all()
    assert hist[nan_pos].isna().all()


# Pandas reference but with NaN positions preserved
def test_macd_matches_pandas_reference():
    prices = pd.Series([1, 2, np.nan, 4, 5, np.nan, np.nan, 8, 9, 10] * 50, dtype=float)
    fast, slow, signal = 12, 26, 9

    macd_line, signal_line, hist = macd(prices, fast=fast, slow=slow, signal=signal)

    ema_fast = (
        prices.ewm(span=fast, adjust=False, ignore_na=True, min_periods=fast)
        .mean()
        .mask(prices.isna())
    )
    ema_slow = (
        prices.ewm(span=slow, adjust=False, ignore_na=True, min_periods=slow)
        .mean()
        .mask(prices.isna())
    )
    macd_ref = (ema_fast - ema_slow).mask(prices.isna())

    signal_ref = (
        macd_ref.ewm(span=signal, adjust=False, ignore_na=True, min_periods=fast + signal)
        .mean()
        .mask(macd_ref.isna())
    )
    hist_ref = (macd_ref - signal_ref).mask(macd_ref.isna() | signal_ref.isna())

    assert_allclose(macd_line.to_numpy(), macd_ref.to_numpy(), equal_nan=True)
    assert_allclose(signal_line.to_numpy(), signal_ref.to_numpy(), equal_nan=True)
    assert_allclose(hist.to_numpy(), hist_ref.to_numpy(), equal_nan=True)
