import numpy as np
import pandas as pd

from mlfinance.validation import validate_prices


def rsi_from_avgs(ag: float, al: float) -> float:
    """Compute RSI from average gain and average loss.

    Parameters:
        ag (float): Average gain.
        al (float): Average loss.

    Returns:
        (float): Relative Strength Index value.
    """
    if al == 0.0:
        return 50.0 if ag == 0.0 else 100.0
    if ag == 0.0:
        return 0.0
    rs = ag / al
    return 100.0 - (100.0 / (1.0 + rs))


def sma(
    x: pd.Series,
    window: int,
    *,
    min_periods: int | None = None,
) -> pd.Series:
    """Compute the Simple Moving Average (SMA).

    Parameters:
        x (pd.Series): Input data series.
        window (int): The moving window size.
        min_periods (int, optional): Minimum number of observations required to have a value.

    Returns:
        (pd.Series): Series containing the SMA values.
    """
    if not isinstance(x, pd.Series):
        raise TypeError("x must be a pandas Series.")
    if window <= 0:
        raise ValueError("window must be positive.")

    mp = window if min_periods is None else min_periods
    return x.rolling(window=window, min_periods=mp).mean()


def rolling_std(
    x: pd.Series,
    window: int,
    *,
    df: int = 0,
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling standard deviation.

    Parameters:
        x (pd.Series): Input data series.
        window (int): The moving window size.
        df (int, optional): Delta degrees of freedom to compute standard deviation, by default 0.
        min_periods (int, optional): Minimum number of observations required to have a value.

    Returns:
        (pd.Series): Series containing the rolling standard deviation values.
    """
    if not isinstance(x, pd.Series):
        raise TypeError("x must be a pandas Series.")
    if window <= 0:
        raise ValueError("window must be positive.")
    if df not in (0, 1):
        raise ValueError("df must be 0 or 1.")

    mp = window if min_periods is None else min_periods
    return x.rolling(window=window, min_periods=mp).std(ddof=df)


def ema_alpha_from_span(span: int) -> float:
    """Compute the alpha parameter for Exponential Moving Average (EMA).

    Parameters:
        span (int): The span (window) for the EMA.

    Returns:
        (float): The alpha parameter for EMA.
    """
    if not isinstance(span, int) or span <= 0:
        raise ValueError("span must be a positive integer.")
    # Allow span >= 1; span=1 => alpha=1
    return 2.0 / (span + 1.0)


def ema(
    prices: pd.Series,
    *,
    span: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Compute the Exponential Moving Average (EMA) of a price series.

    The Exponential Moving Average is a type of Moving Average that grants more
    weight to recent prices. The implementation keeps NaN positions.

    Parameters:
        prices (pd.Series): Series of prices.
        span (int): The span (window) for the EMA.
        min_periods (int, optional): Minimum number of observations required
            to have a value.

    Returns:
        (pd.Series): Series containing the EMA values.

    Sources:
        Investopedia - Exponential Moving Average (EMA)
            https://www.investopedia.com/terms/e/ema.asp
        Pandas documentation - Exponentially weighted window
            https://pandas.pydata.org/docs/reference/api/pandas.Series.ewm.html
    """
    prices = validate_prices(prices)

    if min_periods is None:
        mp = 0
    else:
        if not isinstance(min_periods, int) or min_periods < 0:
            raise ValueError("min_periods must be a non-negative integer or None.")
        mp = min_periods

    a = ema_alpha_from_span(span)

    # Prepare output
    out = pd.Series(np.nan, index=prices.index, dtype=float, name=f"EMA_{span}")

    if prices.empty:
        return out

    x = prices.to_numpy(dtype=float, na_value=np.nan)

    prev = np.nan
    n_valid = 0

    for i, xi in enumerate(x):
        if np.isnan(xi):
            # Output remains NaN
            continue

        n_valid += 1
        prev = xi if np.isnan(prev) else (1.0 - a) * prev + a * xi

        if n_valid >= mp:
            out.iat[i] = prev

    return out
