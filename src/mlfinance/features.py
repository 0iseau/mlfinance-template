"""Feature engineering module.

Provides functions that compute past-only technical indicators and price-volume transformations
used as inputs to machine-learning models.

Functions:
    rsi: Relative Strength Index using Wilder smoothing.
    macd: MACD line, signal line, and histogram.
    bollinger: Bollinger bands.
    obv: On-Balance Volume.
    make_features: Aggregate feature matrix from price and volume data.

Usage:
    X = make_features(df)
    # X contains indicators from price and volume data.
"""

import pandas as pd

# Helper functions


def rsi(Prices: pd.Series[float], window: int) -> pd.Series[float]:
    """Computes the Relative Strength Index (RSI) using Wilder smoothing.

    Args:
        Prices (pd.Series): Series of prices.
        window (int): Lookback period. Default 14.

    Returns:
        pd.Series: RSI values (NaN during warm-up or where Prices contains NaN).
    """
    # Empty series → return empty series (clean)
    if Prices.empty:
        return pd.Series(index=Prices.index, dtype=float)

    # Diff preserves NaNs naturally
    delta = Prices.diff()

    # Gains and losses preserve NaN alignment
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Prepare outputs
    avg_gain = pd.Series(index=Prices.index, dtype=float)
    avg_loss = pd.Series(index=Prices.index, dtype=float)

    # If not enough data for warm-up → return all NaN
    if len(Prices) <= window:
        return pd.Series(index=Prices.index, dtype=float)

    # Initial average gain/loss (includes NaNs if present)
    initial_slice = gain.iloc[1 : window + 1]
    avg_gain.iloc[window] = initial_slice.mean()
    avg_loss.iloc[window] = loss.iloc[1 : window + 1].mean()

    # Wilder smoothing
    for i in range(window + 1, len(Prices)):
        # If the current price or delta is NaN → keep previous averages
        if pd.isna(gain.iloc[i]) or pd.isna(loss.iloc[i]):
            avg_gain.iloc[i] = avg_gain.iloc[i - 1]
            avg_loss.iloc[i] = avg_loss.iloc[i - 1]
            continue

        avg_gain.iloc[i] = ((avg_gain.iloc[i - 1] * (window - 1)) + gain.iloc[i]) / window
        avg_loss.iloc[i] = ((avg_loss.iloc[i - 1] * (window - 1)) + loss.iloc[i]) / window

    # Compute RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Constant-series case: avg_gain = avg_loss = 0 → RSI = 50
    mask_constant = (avg_gain == 0) & (avg_loss == 0)
    rsi[mask_constant] = 50

    # FINAL STEP: preserve NaNs from the original input
    rsi[Prices.isna()] = float("nan")

    return rsi
