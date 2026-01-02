import numpy as np
import pandas as pd
import pytest

import mlfinance.features as ft


def test_obv_preserves_index_and_length():
    idx = pd.RangeIndex(5)
    close = pd.Series([10, 11, 10, 10, 12], index=idx, dtype=float)
    volume = pd.Series([100, 200, 150, 300, 250], index=idx, dtype=float)

    out = ft.obv(close, volume)

    assert isinstance(out, pd.Series)
    assert out.index.equals(idx)
    assert len(out) == len(close)


def test_obv_simple_computation():
    close = pd.Series([10.0, 11.0, 10.0, 10.0, 12.0])
    volume = pd.Series([100.0, 200.0, 150.0, 300.0, 250.0])

    out = ft.obv(close, volume)

    expected = pd.Series([0.0, 200.0, 50.0, 50.0, 300.0], index=close.index)

    pd.testing.assert_series_equal(out, expected, check_names=False)


def test_obv_flat_prices_gives_constant_obv():
    close = pd.Series([10.0, 10.0, 10.0, 10.0])
    volume = pd.Series([100.0, 200.0, 300.0, 400.0])

    out = ft.obv(close, volume)

    expected = pd.Series([0.0, 0.0, 0.0, 0.0], index=close.index)
    pd.testing.assert_series_equal(out, expected, check_names=False)


def test_obv_raises_on_index_mismatch():
    close = pd.Series([10.0, 11.0, 12.0], index=[0, 1, 2])
    volume = pd.Series([100.0, 200.0, 300.0], index=[1, 2, 3])

    with pytest.raises(ValueError):
        ft.obv(close, volume)


def test_obv_raises_on_non_positive_close():
    close = pd.Series([10.0, 0.0, 12.0], dtype=float)
    volume = pd.Series([100.0, 200.0, 300.0], dtype=float)

    with pytest.raises(ValueError):
        ft.obv(close, volume)


def test_obv_raises_on_negative_volume():
    close = pd.Series([10.0, 11.0, 12.0], dtype=float)
    volume = pd.Series([100.0, -200.0, 300.0], dtype=float)

    with pytest.raises(ValueError):
        ft.obv(close, volume)


def test_obv_nan_propagation_at_nan_timestamps():
    close = pd.Series([10.0, np.nan, 12.0, 11.0])
    volume = pd.Series([100.0, 200.0, np.nan, 400.0])

    out = ft.obv(close, volume)

    # NaN where either input is NaN
    assert np.isnan(out.iloc[1])
    assert np.isnan(out.iloc[2])


def test_obv_matches_ta_library_on_non_ties():
    try:
        from ta.volume import OnBalanceVolumeIndicator
    except ImportError:
        pytest.fail("Package 'ta' is required for this test. Install with: uv add --dev ta")

    close = pd.Series([10.0, 11.0, 10.0, 12.0, 11.0, 13.0])  # no ties
    volume = pd.Series([100.0, 200.0, 150.0, 250.0, 400.0, 500.0])

    ours = ft.obv(close, volume)
    ref = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()

    # Align initial level (ta seeds differently)
    ref = ref - ref.iloc[0]

    assert np.allclose(ours.to_numpy(), ref.to_numpy(), rtol=1e-12, atol=1e-12)
