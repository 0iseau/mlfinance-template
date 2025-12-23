import numpy as np
import pandas as pd
import pytest

import mlfinance.features as ft


def test_atr_returns_series_with_same_index():
    idx = pd.RangeIndex(6)
    high = pd.Series([10, 11, 12, 13, 14, 15], index=idx, dtype=float)
    low = pd.Series([9, 10, 11, 12, 13, 14], index=idx, dtype=float)
    close = pd.Series([9.5, 10.5, 11.5, 12.5, 13.5, 14.5], index=idx, dtype=float)

    out = ft.atr(high, low, close, window=3)

    assert isinstance(out, pd.Series)
    assert out.index.equals(idx)
    assert len(out) == len(close)


def test_atr_is_nan_until_seed_window():
    high = pd.Series([10, 11, 12, 13, 14], dtype=float)
    low = pd.Series([9, 10, 11, 12, 13], dtype=float)
    close = pd.Series([9.5, 10.5, 11.5, 12.5, 13.5], dtype=float)

    out = ft.atr(high, low, close, window=3)

    # avec Wilder: seed au 3e TR valide -> NaN avant
    assert out.iloc[:2].isna().all()
    assert np.isfinite(out.iloc[2])


def test_true_range_is_constant_in_constructed_data():
    close = pd.Series([10, 10, 10, 10, 10, 10], dtype=float)
    high = close + 0.5
    low = close - 0.5

    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(
        axis=1
    )

    assert np.allclose(tr.dropna().to_numpy(), 1.0)


def test_atr_constant_true_range_gives_constant_atr_after_seed():
    close = pd.Series([10, 10, 10, 10, 10, 10], dtype=float)
    high = close + 0.5
    low = close - 0.5  # TR = 1 always

    out = ft.atr(high, low, close, window=3)
    tail = out.dropna()

    assert np.allclose(tail.to_numpy(), 1.0)


def test_atr_raises_on_invalid_window():
    high = pd.Series([10, 11, 12], dtype=float)
    low = pd.Series([9, 10, 11], dtype=float)
    close = pd.Series([9.5, 10.5, 11.5], dtype=float)

    with pytest.raises(ValueError):
        ft.atr(high, low, close, window=0)

    with pytest.raises(TypeError):
        ft.atr(high, low, close, window=2.5)


def test_atr_raises_on_index_mismatch():
    high = pd.Series([10, 11, 12], index=[0, 1, 2], dtype=float)
    low = pd.Series([9, 10, 11], index=[0, 1, 2], dtype=float)
    close = pd.Series([9.5, 10.5, 11.5], index=[1, 2, 3], dtype=float)

    with pytest.raises(ValueError):
        ft.atr(high, low, close, window=2)


def test_true_range_matches_talib_if_available():
    talib = pytest.importorskip("talib")

    high = pd.Series([10, 11, 10.8, 12.2, 12.0, 13.0], dtype=float)
    low = pd.Series([9.5, 10.2, 10.1, 11.4, 11.2, 12.1], dtype=float)
    close = pd.Series([9.8, 10.9, 10.3, 12.0, 11.5, 12.8], dtype=float)

    ours_tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1
    ).max(axis=1)

    ref_tr = pd.Series(
        talib.TRANGE(high.to_numpy(), low.to_numpy(), close.to_numpy()),
        index=close.index,
        dtype=float,
    )

    mask = ours_tr.notna() & ref_tr.notna()
    assert mask.any()
    assert np.allclose(ours_tr[mask].to_numpy(), ref_tr[mask].to_numpy(), rtol=1e-12, atol=1e-12)
