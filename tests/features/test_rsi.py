import pandas as pd

from mlfinance.features import rsi


def test_rsi_warmup(price_series_inc) -> None:
    """Test that the RSI function produces NaN values for the warm-up period,
    and valid values afterwards."""
    window = 14
    s = rsi(price_series_inc, window)
    assert s.iloc[:window].isna().all()
    assert s.iloc[window:].notna().all()


def test_rsi_type(price_series_inc) -> None:
    """Test that the RSI function returns a pd.Series."""
    window = 14
    rsi_series = rsi(price_series_inc, window)
    assert isinstance(rsi_series, pd.Series)


def test_rsi_length(price_series_inc) -> None:
    """Test that the RSI function returns a Series of the same length as the input."""
    window = 14
    rsi_series = rsi(price_series_inc, window)
    assert len(rsi_series) == len(price_series_inc)


def test_rsi_values_range(price_series_random) -> None:
    """Test that the RSI values are within the expected range [0, 100], ignoring NaNs."""
    window = 14
    rsi_series = rsi(price_series_random, window)
    assert rsi_series.dropna().between(0, 100).all()


def test_rsi_no_lookahead(price_series_random) -> None:
    """RSI should change if we shift the price series in time."""
    window = 14
    s = price_series_random

    rsi_original = rsi(s, window)
    rsi_shifted = rsi(s.shift(1).bfill(), window)

    part_original = rsi_original.iloc[window : window + 20]
    part_shifted = rsi_shifted.iloc[window : window + 20]

    assert not part_original.equals(part_shifted)


def test_rsi_increasing_series(price_series_inc) -> None:
    """Test that the RSI function returns high values (>50) for an increasing price series."""
    window = 14
    rsi_series = rsi(price_series_inc, window)
    assert (rsi_series.iloc[window:] > 50).all()


def test_rsi_decreasing_series(price_series_dec) -> None:
    """Test that the RSI function returns low values (<50) for a decreasing price series."""
    window = 14
    rsi_series = rsi(price_series_dec, window)
    assert (rsi_series.iloc[window:] < 50).all()


def test_rsi_constant_series(price_series_constant) -> None:
    """Test that the RSI function returns 50 for a constant price series after warm-up."""
    window = 14
    rsi_series = rsi(price_series_constant, window)
    assert rsi_series.iloc[window:].eq(50).all()


def test_rsi_empty_series() -> None:
    """Test that the RSI function handles an empty Series without error."""
    empty_series = pd.Series(dtype=float)
    rsi_series = rsi(empty_series, window=14)
    assert rsi_series.empty


def test_rsi_preserves_nans():
    """Test that the RSI function preserves NaN values in the input Series."""
    prices = pd.Series([10, 11, None, 13, 14], dtype=float)
    out = rsi(prices, 14)
    assert out.isna().iloc[2]
