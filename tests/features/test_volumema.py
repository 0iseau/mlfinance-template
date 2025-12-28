import numpy as np
import pandas as pd

from mlfinance.features import volume_ma
from mlfinance.math_utils import sma


def test_volume_ma_basic_properties():
    idx = pd.date_range("2025-01-01", periods=7, freq="D")
    volume = pd.Series([100, 120, 90, 110, 130, 80, 140], index=idx)

    vma = volume_ma(volume, 3)

    assert vma.index.equals(volume.index)
    assert vma.dtype == float
    assert vma.isna().sum() == 2


def test_volume_ma_matches_public_reference():
    idx = pd.date_range("2025-01-01", periods=7, freq="D")
    volume = pd.Series([5, 7, 6, 8, 10, 9, 11], index=idx)

    expected = volume.rolling(3).mean().astype(float)
    expected.name = "VMA_3"

    result = volume_ma(volume, 3)

    assert np.allclose(
        result.to_numpy(),
        expected.to_numpy(),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )


def test_volume_ma_vs_pandas_sma_function():
    idx = pd.date_range("2025-01-01", periods=6, freq="D")
    volume = pd.Series([20, 30, 25, 35, 40, 45], index=idx)

    yours = volume_ma(volume, 4)
    reference = sma(volume, 4)

    assert np.allclose(
        yours.to_numpy(),
        reference.to_numpy(),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
