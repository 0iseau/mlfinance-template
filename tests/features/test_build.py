from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure "src/" is importable when running tests without an installed package.
_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

from mlfinance.features import (  # noqa: E402
    atr,
    bollinger_bands,
    build_feature,
    macd,
    obv,
    returns,
    rolling_indicators,
    rolling_volatility,
    rsi,
    rvol,
    volatility,
    volume_ma,
)


@pytest.fixture()
def sample_ohlcv() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=12, freq="D")
    close = pd.Series(
        [100.0, 101.0, 102.0, 104.0, 103.0, 105.0, 106.0, 108.0, 107.0, 109.0, 110.0, 111.0],
        index=idx,
        name="close",
    )
    return pd.DataFrame(
        {
            "close": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "volume": np.arange(1, len(close) + 1, dtype=float) * 100.0,
        },
        index=idx,
    )


# -----------------------------
# Core robustness expectations
# -----------------------------


def test_build_feature_does_not_mutate_input(sample_ohlcv: pd.DataFrame) -> None:
    df_in = sample_ohlcv.copy(deep=True)
    _ = build_feature(df_in, option="analyze")
    pd.testing.assert_frame_equal(df_in, sample_ohlcv)


@pytest.mark.parametrize("option", ["analyze", "summary", "lags"])
def test_build_feature_preserves_index(sample_ohlcv: pd.DataFrame, option: str) -> None:
    got = build_feature(sample_ohlcv, option=option)
    assert got is not sample_ohlcv
    assert got.index.equals(sample_ohlcv.index) or option == "summary"
    if option == "summary":
        # Summary is typically 1-row; allow either RangeIndex or same index,
        # but enforce "single observation" contract.
        assert len(got) == 1


def test_build_feature_invalid_option_raises(sample_ohlcv: pd.DataFrame) -> None:
    # Your implementation raises TypeError for invalid option
    with pytest.raises(TypeError):
        _ = build_feature(sample_ohlcv, option="not_a_real_option")


# -----------------------------
# Analyze mode: schema + spot checks
# -----------------------------


def test_build_feature_analyze_schema(sample_ohlcv: pd.DataFrame) -> None:
    got = build_feature(sample_ohlcv, option="analyze")

    # Minimal schema guarantees: these are the "core" signals you're wiring.
    expected_cols = {
        "rsi",
        "macd_line",
        "signal_line",
        "macd_hist",
        "bb_mid",
        "bb_upper",
        "bb_lower",
        "returns",
        "volatility",
        "rolling_volatility",
        "atr",
        "obv",
        "volume_ma",
        "rvol",
        "roll_mean_10",
        "roll_std_10",
        "roll_min_10",
        "roll_max_10",
    }
    missing = expected_cols - set(got.columns)
    assert not missing, f"Missing expected columns in analyze output: {sorted(missing)}"

    # Shape must match input length (analyze produces time-aligned features)
    assert len(got) == len(sample_ohlcv)
    assert got.index.equals(sample_ohlcv.index)


def test_build_feature_analyze_spot_checks(sample_ohlcv: pd.DataFrame) -> None:
    got = build_feature(sample_ohlcv, option="analyze")

    price = sample_ohlcv["close"].astype(float)
    high = sample_ohlcv["high"].astype(float)
    low = sample_ohlcv["low"].astype(float)
    vol = sample_ohlcv["volume"].astype(float)

    # RSI spot-check
    exp_rsi = rsi(price)
    pd.testing.assert_series_equal(
        got["rsi"],
        exp_rsi,
        check_names=False,
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )

    # MACD spot-check
    exp_macd_line, exp_signal, exp_hist = macd(price)
    pd.testing.assert_series_equal(
        got["macd_line"], exp_macd_line, check_names=False, check_dtype=False
    )
    pd.testing.assert_series_equal(
        got["signal_line"], exp_signal, check_names=False, check_dtype=False
    )
    pd.testing.assert_series_equal(got["macd_hist"], exp_hist, check_names=False, check_dtype=False)

    # Bollinger spot-check (avoid .T.values brittleness; select by names if present)
    bb = bollinger_bands(price)
    # Allow either your own naming or a conventional naming; map if needed.
    # Preferred: bb has columns ["bb_mid","bb_upper","bb_lower"].
    if {"bb_mid", "bb_upper", "bb_lower"}.issubset(set(bb.columns)):
        pd.testing.assert_series_equal(
            got["bb_mid"], bb["bb_mid"], check_names=False, check_dtype=False
        )
        pd.testing.assert_series_equal(
            got["bb_upper"], bb["bb_upper"], check_names=False, check_dtype=False
        )
        pd.testing.assert_series_equal(
            got["bb_lower"], bb["bb_lower"], check_names=False, check_dtype=False
        )

    # Returns spot-check (ret_1)
    exp_ret1 = returns(price)["ret_1"]
    pd.testing.assert_series_equal(got["returns"], exp_ret1, check_names=False, check_dtype=False)

    # ATR spot-check
    exp_atr = atr(high, low, price)
    pd.testing.assert_series_equal(got["atr"], exp_atr, check_names=False, check_dtype=False)

    # OBV spot-check
    exp_obv = obv(price, vol)
    pd.testing.assert_series_equal(got["obv"], exp_obv, check_names=False, check_dtype=False)

    # Rolling indicators spot-check (10)
    roll = rolling_indicators(price, window=10)
    pd.testing.assert_series_equal(
        got["roll_mean_10"], roll["roll_mean_10"], check_names=False, check_dtype=False
    )
    pd.testing.assert_series_equal(
        got["roll_std_10"], roll["roll_std_10"], check_names=False, check_dtype=False
    )
    pd.testing.assert_series_equal(
        got["roll_min_10"], roll["roll_min_10"], check_names=False, check_dtype=False
    )
    pd.testing.assert_series_equal(
        got["roll_max_10"], roll["roll_max_10"], check_names=False, check_dtype=False
    )

    # Rolling volatility spot-check (21)
    exp_rv = rolling_volatility(price, window=21)
    pd.testing.assert_series_equal(
        got["rolling_volatility"], exp_rv, check_names=False, check_dtype=False
    )

    # Volatility contract: if volatility() is scalar, build_feature should broadcast
    v = volatility(price)
    if np.isscalar(v):
        assert np.allclose(got["volatility"].to_numpy(), float(v), rtol=0.0, atol=0.0)
    else:
        pd.testing.assert_series_equal(got["volatility"], v, check_names=False, check_dtype=False)

    # Volume MA and RVOL spot-checks
    pd.testing.assert_series_equal(
        got["volume_ma"], volume_ma(vol, window=20), check_names=False, check_dtype=False
    )
    pd.testing.assert_series_equal(
        got["rvol"], rvol(vol, window=20), check_names=False, check_dtype=False
    )


# -----------------------------
# Summary mode: correctness against pandas
# -----------------------------


def test_build_feature_summary_values(sample_ohlcv: pd.DataFrame) -> None:
    got = build_feature(sample_ohlcv, option="summary")

    price = sample_ohlcv["close"].astype(float)
    vol = sample_ohlcv["volume"].astype(float)

    # Enforce one-row summary (common CLI behavior)
    assert len(got) == 1

    # Minimal expected columns (extend if your summary includes more)
    expected_cols = {
        "price_mean",
        "price_std",
        "price_var",
        "min_price",
        "max_price",
        "median_price",
        "volume_mean",
        "volume_std",
        "volume_var",
        "min_volume",
        "max_volume",
    }
    missing = expected_cols - set(got.columns)
    assert not missing, f"Missing expected columns in summary output: {sorted(missing)}"

    row = got.iloc[0]

    # pandas defaults: std/var are ddof=1 unless you set otherwise in your implementation.
    assert np.isclose(row["price_mean"], price.mean(), rtol=1e-12, atol=1e-12)
    assert np.isclose(row["price_std"], price.std(), rtol=1e-12, atol=1e-12)
    assert np.isclose(row["price_var"], price.var(), rtol=1e-12, atol=1e-12)
    assert np.isclose(row["min_price"], price.min(), rtol=0.0, atol=0.0)
    assert np.isclose(row["max_price"], price.max(), rtol=0.0, atol=0.0)
    assert np.isclose(row["median_price"], price.median(), rtol=0.0, atol=0.0)

    assert np.isclose(row["volume_mean"], vol.mean(), rtol=1e-12, atol=1e-12)
    assert np.isclose(row["volume_std"], vol.std(), rtol=1e-12, atol=1e-12)
    assert np.isclose(row["volume_var"], vol.var(), rtol=1e-12, atol=1e-12)
    assert np.isclose(row["min_volume"], vol.min(), rtol=0.0, atol=0.0)
    assert np.isclose(row["max_volume"], vol.max(), rtol=0.0, atol=0.0)


# -----------------------------
# Lag mode: robust checks without over-assuming exact naming
# -----------------------------


def _extract_lag_k(col: str) -> int | None:
    # Accept patterns like: "lag_1", "lag1", "Price_lag_5", "price_lag5", etc.
    m = re.search(r"lag[_]?(?P<k>\d+)", col, flags=re.IGNORECASE)
    if not m:
        return None
    return int(m.group("k"))


def test_build_feature_lag_creates_shifted_columns(sample_ohlcv: pd.DataFrame) -> None:
    got = build_feature(sample_ohlcv, option="lags")
    assert len(got) == len(sample_ohlcv)
    assert got.index.equals(sample_ohlcv.index)

    # Must contain at least one lag column
    lag_cols = [c for c in got.columns if re.search(r"lag", c, flags=re.IGNORECASE)]
    assert lag_cols, "Lag mode produced no columns containing 'lags'"

    price = sample_ohlcv["close"].astype(float)

    # For any column where we can parse k from the name, verify it's a shift(k) of price.
    checked_any = False
    for c in lag_cols:
        k = _extract_lag_k(c)
        if k is None:
            continue
        checked_any = True
        exp = price.shift(k)
        pd.testing.assert_series_equal(got[c], exp, check_names=False, check_dtype=False)

    # If naming doesn't encode k, still enforce non-triviality:
    # at least one column differs from price.
    if not checked_any:
        assert any(
            not got[c].equals(price) for c in lag_cols
        ), "Lag columns exist but none appear to be shifted versions of price"


# -----------------------------
# Error handling: missing columns
# -----------------------------


def test_build_feature_missing_price_column_raises(sample_ohlcv: pd.DataFrame) -> None:
    df = sample_ohlcv.drop(columns=["close"])
    # Current implementation raises StopIteration (from next(...))
    with pytest.raises(StopIteration):
        _ = build_feature(df, option="analyze")


def test_build_feature_missing_volume_column_summary_raises_or_nans(
    sample_ohlcv: pd.DataFrame,
) -> None:
    df = sample_ohlcv.drop(columns=["volume"])
    # Current implementation raises StopIteration (from next(...))
    with pytest.raises(StopIteration):
        _ = build_feature(df, option="summary")
