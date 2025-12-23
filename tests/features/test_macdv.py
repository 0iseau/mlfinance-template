import numpy as np
import pandas as pd
import pytest

import mlfinance.features as ft


def test_macd_v_returns_series_with_same_index_and_length():
    idx = pd.RangeIndex(60)
    close = pd.Series(np.linspace(100, 130, len(idx)), index=idx, dtype=float)
    high = close + 1.0
    low = close - 1.0

    out = ft.macd_v(close, high, low, fast=3, slow=6, atr_window=6)

    assert isinstance(out, pd.Series)
    assert out.index.equals(close.index)
    assert len(out) == len(close)


def test_macd_v_is_nan_when_atr_is_nan():
    # For short series, ATR (Wilder) will be NaN until seeded => macd_v should be NaN there.
    close = pd.Series([100, 101, 102, 103, 104], dtype=float)
    high = close + 1.0
    low = close - 1.0

    out = ft.macd_v(close, high, low, fast=2, slow=3, atr_window=10)

    assert out.isna().all()


def test_macd_v_matches_manual_formula_using_pandas_ewm_and_our_atr():
    close = pd.Series([100, 101, 99, 102, 101, 103, 104, 102, 105, 106], dtype=float)
    high = close + 1.0
    low = close - 1.0

    fast, slow, atr_window = 3, 6, 3
    out = ft.macd_v(close, high, low, fast=fast, slow=slow, atr_window=atr_window)

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    atr_val = ft.atr(high, low, close, window=atr_window)

    expected = macd / atr_val

    # Compare where both are defined
    mask = out.notna() & expected.notna()
    assert mask.any()
    assert np.allclose(out[mask].to_numpy(), expected[mask].to_numpy(), rtol=1e-12, atol=1e-12)


def test_macd_v_raises_on_invalid_parameters():
    close = pd.Series([100, 101, 102], dtype=float)
    high = close + 1.0
    low = close - 1.0

    with pytest.raises(ValueError):
        ft.macd_v(close, high, low, fast=0, slow=26, atr_window=26)

    with pytest.raises(ValueError):
        ft.macd_v(close, high, low, fast=12, slow=-1, atr_window=26)

    with pytest.raises(ValueError):
        ft.macd_v(close, high, low, fast=12, slow=26, atr_window=0)


def test_macd_v_raises_on_index_mismatch():
    close = pd.Series([100, 101, 102], index=[0, 1, 2], dtype=float)
    high = pd.Series([101, 102, 103], index=[0, 1, 2], dtype=float)
    low = pd.Series([99, 100, 101], index=[1, 2, 3], dtype=float)

    with pytest.raises(ValueError):
        ft.macd_v(close, high, low, fast=2, slow=3, atr_window=2)
