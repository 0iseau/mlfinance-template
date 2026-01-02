import numpy as np
import pandas as pd
import pytest

from mlfinance.features import rvol


def test_rvol_basic_properties():
    idx = pd.date_range("2025-01-01", periods=7, freq="D")
    volume = pd.Series([10, 20, 30, 40, 50, 60, 70], index=idx)

    out = rvol(volume, window=3)

    assert isinstance(out, pd.Series)
    assert out.index.equals(volume.index)
    assert out.dtype == float
    assert out.shape == volume.shape


def test_rvol_has_expected_initial_nans():
    idx = pd.date_range("2025-01-01", periods=6, freq="D")
    volume = pd.Series([5, 5, 5, 5, 5, 5], index=idx)

    out = rvol(volume, window=3)

    # With min_periods=window, first window-1 values are NaN
    assert out.iloc[:2].isna().all()


def test_rvol_invalid_window_raises():
    idx = pd.date_range("2025-01-01", periods=3, freq="D")
    volume = pd.Series([1.0, 2.0, 3.0], index=idx)

    with pytest.raises(ValueError):
        rvol(volume, window=0)

    with pytest.raises(ValueError):
        rvol(volume, window=-1)

    with pytest.raises(ValueError):
        rvol(volume, window=2.5)  # type: ignore[arg-type]


def test_rvol_raises_on_negative_volume():
    volume = pd.Series([10.0, -2.0, 5.0], dtype=float)
    with pytest.raises(ValueError):
        rvol(volume, window=2)


def test_rvol_empty_input_returns_empty_series():
    volume = pd.Series([], dtype=float)

    out = rvol(volume, window=3)

    assert isinstance(out, pd.Series)
    assert out.index.equals(volume.index)
    assert out.shape == volume.shape


def test_rvol_matches_public_reference_pandas():
    # Public reference: pandas rolling mean implementation
    idx = pd.date_range("2025-01-01", periods=8, freq="D")
    volume = pd.Series([10, 12, 11, 15, 14, 16, 18, 17], index=idx).astype(float)

    window = 4
    out = rvol(volume, window=window)

    sma = volume.rolling(window=window, min_periods=window).mean()
    expected = (volume / sma).replace([np.inf, -np.inf], np.nan).astype(float)

    assert np.allclose(
        out.to_numpy(),
        expected.to_numpy(),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
