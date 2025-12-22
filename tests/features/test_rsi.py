import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from mlfinance.features import rsi


def test_rsi_preserves_index_and_length():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    s = pd.Series(np.linspace(100, 110, 10), index=idx)
    out = rsi(s, window=3)
    assert out.index.equals(s.index)
    assert len(out) == len(s)


def test_rsi_returns_series():
    s = pd.Series([1, 2, 3, 2, 2, 5], dtype=float)
    out = rsi(s, window=3)
    assert isinstance(out, pd.Series)


@pytest.mark.parametrize("window", [None, 1.5, "14"])
def test_rsi_window_type_error(window):
    s = pd.Series([1, 2, 3], dtype=float)
    with pytest.raises(TypeError):
        rsi(s, window=window)


@pytest.mark.parametrize("window", [0, -1])
def test_rsi_window_value_error(window):
    s = pd.Series([1, 2, 3], dtype=float)
    with pytest.raises(ValueError):
        rsi(s, window=window)


def test_rsi_empty_series_returns_empty():
    s = pd.Series([], dtype=float)
    out = rsi(s, window=14)
    assert len(out) == 0
    assert out.name == "RSI_14"


def test_rsi_leading_nans_until_window():
    s = pd.Series(np.arange(1, 30, dtype=float))
    out = rsi(s, window=14)
    assert out.iloc[:14].isna().all()
    assert not np.isnan(out.iloc[14])


def test_rsi_block_len_le_window_is_all_nan():
    s = pd.Series([10, 11, 12], dtype=float)
    out = rsi(s, window=3)
    assert out.isna().all()


def test_rsi_blocks_restart_and_nan_preserved():
    # bloc 1 of length 5 NaN on index 5, bloc 2 of length 6
    s = pd.Series([1, 2, 3, 4, 5, np.nan, 10, 11, 12, 13, 14, 15], dtype=float)
    w = 3
    out = rsi(s, window=w)

    # NaN input stays NaN
    assert np.isnan(out.iloc[5])

    # bloc 1 start = 0 -> out[0..2] NaN, out[3] first defined
    assert out.iloc[0:w].isna().all()
    assert not np.isnan(out.iloc[w])

    # bloc 2 start = 6 -> out[6..8] NaN, out[9] first defined
    assert out.iloc[6 : 6 + w].isna().all()
    assert not np.isnan(out.iloc[6 + w])


def test_rsi_bounded_0_100_on_valid_points():
    s = pd.Series(np.linspace(100, 120, 50), dtype=float)
    out = rsi(s, window=14).dropna()
    assert ((out >= 0) & (out <= 100)).all()


def test_rsi_sets_name_and_dtype():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    out = rsi(s, window=3)
    assert out.name == "RSI_3"
    assert out.dtype == float


def test_rsi_preserves_nan_positions():
    s = pd.Series([1, 2, np.nan, 3, 4], dtype=float)
    out = rsi(s, window=2)
    assert np.isnan(out.iloc[2])


def test_rsi_monotone_increasing_hits_upper_bound():
    s = pd.Series(np.arange(1, 40, dtype=float))
    out = rsi(s, window=14).dropna()
    assert_allclose(out.to_numpy(), np.full_like(out, 100.0), rtol=1e-7, atol=1e-7)


def test_rsi_with_constant_prices():
    s = pd.Series([100] * 50, dtype=float)
    out = rsi(s, window=14).dropna()
    assert_allclose(out.to_numpy(), 50.0)


def test_rsi_matches_talib_wilder():
    ta = pytest.importorskip("talib")

    s = pd.Series(np.linspace(100, 120, 300), dtype=float)
    window = 14

    ours = rsi(s, window=window)
    ref = ta.RSI(s.to_numpy(), timeperiod=window)

    mask = ~np.isnan(ref)
    assert_allclose(
        ours[mask].to_numpy(),
        ref[mask],
        rtol=1e-8,
        atol=1e-8,
    )


def test_rsi_matches_talib_wilder_complex_series():
    pytest.importorskip("talib")
    import talib as ta

    rng = np.random.default_rng(0)
    returns = rng.normal(loc=0.0002, scale=0.01, size=500)
    prices = 100 * np.exp(np.cumsum(returns))
    s = pd.Series(prices, dtype=float)

    window = 14
    ours = rsi(s, window=window)
    ref = ta.RSI(s.to_numpy(), timeperiod=window)

    mask = ~np.isnan(ref)
    assert_allclose(ours[mask].to_numpy(), ref[mask], rtol=1e-8, atol=1e-8)
