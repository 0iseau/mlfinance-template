"""Feature engineering module.

Provides functions that compute technical indicators and price-volume transformations
used as inputs to machine-learning models.

Functions:
    Technical indicators:
        rsi: Relative Strength Index using Wilder smoothing.
        macd: MACD line, signal line, and histogram.
        bollinger: Bollinger bands.
    Price-based features:
        returns: Simple and log rate of returns.
        volatility:
        momentum:
        trend:
    Volume-based features:
        obv: On-Balance Volume.
        vma: Volume Moving Average.
        volume volatility:
    Lag and rolling statistics:
        lagged values of prices and volume.
        rolling mean, std, min, max of prices and volume.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

import mlfinance.math_utils as mu
from mlfinance.validation import validate_prices


def rsi(prices: pd.Series[float], window: int = 14) -> pd.Series[float]:
    """Relative Strength Index (RSI).

    The RSI is a momentum oscillator bounded from 0 to 100 that measures the strength of
    recent closing-price changes over a specified trading period. It is useful to identify the
    strength or weakness of a market.

    Calculation:
        Default input:
            window = 14
        RS = Average Gain over window period / Average Loss over window period
        RSI = 100 - (100 / (1 + RS))

    Parameters:
        prices (pd.Series[float]): Series of closing prices.
        window (int, optional): Number of periods to use for calculating the RSI, by default 14.

    Returns:
        rsi (pd.Series[float]): Series containing the RSI values.

    Sources:
        Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*
        Murphy, J. J. (1999). *Technical Analysis of the Financial Markets*
        TradingView, “Relative Strength Index (RSI)”.
            https://www.tradingview.com/support/solutions/43000502338-relative-strength-index-rsi/
    """
    # error handling
    prices = validate_prices(prices)

    if not isinstance(window, int):
        raise TypeError("window must be an integer.")
    if window <= 0:
        raise ValueError("window must be a positive integer.")

    # prefill output with NaNs
    out = pd.Series(
        np.nan,
        index=prices.index,
        dtype=float,
        name=f"RSI_{window}",
    )

    # convert to numpy array for speed and split data into valid blocks (without NaNs).
    # RSI stays NaN over NaN regions
    x = prices.to_numpy(dtype=float, na_value=np.nan)
    n = x.size
    if n == 0:
        return out

    isnan = np.isnan(x)

    # Process each contiguous block of non-NaN prices independently
    start = None
    for i in range(n + 1):
        if i < n and not isnan[i]:
            if start is None:
                start = i
        else:
            if start is None:
                continue
            end = i  # exclusive
            block_len = end - start
            if block_len > window:
                block = x[start:end]
                delta = np.diff(block)

                # split gains and losses periods [+2, -1, +3] -> gain=[2,0,3], loss=[0,1,0]
                gain = np.where(delta > 0.0, delta, 0.0)
                loss = np.where(delta < 0.0, -delta, 0.0)

                # Seed with simple mean over first `window` changes
                avg_gain = gain[:window].mean()
                avg_loss = loss[:window].mean()

                # First RSI value lands at index (start + window)
                out.iloc[start + window] = mu.rsi_from_avgs(avg_gain, avg_loss)

                # Wilder smoothing, delta[j] ends at price index (start+j+1)
                for j in range(window, delta.size):
                    avg_gain = (avg_gain * (window - 1) + gain[j]) / window
                    avg_loss = (avg_loss * (window - 1) + loss[j]) / window
                    out.iloc[start + j + 1] = mu.rsi_from_avgs(avg_gain, avg_loss)

            start = None

    return out


def macd(
    prices: pd.Series,
    *,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute the Moving Average Convergence Divergence (MACD) indicator.

    The MACD computes the difference between a fast and a slow Exponential Moving Average (EMA)
    of prices. It measures the tendency, momentum and strength of a financial asset.
    The implementation keeps NaN positions.

    Calculation:
        Default inputs:
            fast = 12
            slow = 26
            signal = 9
        MACD Line = EMA_fast(prices) - EMA_slow(prices)
        Signal Line = EMA_signal(MACD Line)
        MACD Histogram = MACD Line - Signal Line

    Parameters:
        prices (pd.Series): Series of closing prices.
        fast (int, optional): Span for the fast EMA, by default 12.
        slow (int, optional): Span for the slow EMA, by default 26.
        signal (int, optional): Span for the signal line EMA, by default 9.

    Returns:
        macd_line (pd.Series): Series containing the MACD line values.
        signal_line (pd.Series): Series containing the signal line values.
        hist (pd.Series): Series containing the MACD histogram values.

    Sources:
        Appel, G. (2005). *Technical Analysis: Power Tools for Active Investors.*
        Investopedia - Moving Average Convergence Divergence (MACD)
            https://www.investopedia.com/terms/m/macd.asp
    """
    prices = validate_prices(prices)

    # parameter verifications
    for name, v in (("fast", fast), ("slow", slow), ("signal", signal)):
        if not isinstance(v, int) or v <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    if fast >= slow:
        raise ValueError("fast must be < slow.")

    ema_fast = mu.ema(prices, span=fast, min_periods=fast)
    ema_slow = mu.ema(prices, span=slow, min_periods=slow)

    macd_line = (ema_fast - ema_slow).astype(float)
    macd_line.name = f"MACD_{fast}_{slow}"

    signal_line = mu.ema(macd_line, span=signal, min_periods=fast + signal)
    signal_line.name = f"MACDsig_{fast}_{slow}_{signal}"

    hist = (macd_line - signal_line).astype(float)
    hist.name = f"MACDhist_{fast}_{slow}_{signal}"

    return macd_line, signal_line, hist


def bollinger_bands(
    close: pd.Series,
    window: int = 20,
    n_std: float = 2.0,
) -> pd.DataFrame:
    """Bollinger Bands.

    Tool to assess the volatility of stocks and other securities, to identify
    over-valuation or under-valuation. Consists of a center band (Simple Moving Average)
    and lower and upper bands directed by specific standard deviations.

    Parameters:
        close (pd.Series): Series of closing prices.
        window (int, optional): Number of periods to use for calculating the bands, by default 20.
        n_std (float, optional): Number of standard deviations for the upper and lower bands,
            by default 2.0.
        ddof (int, optional): Delta degrees of freedom to compute standard deviation, by default 0.
        min_periods (int | None, optional): Minimum number of observations required to have a value,
            by default None.

    Returns:
        pd.DataFrame: DataFrame containing the Bollinger Bands features

    Sources:
        Bollinger, J. (2002). *Bollinger on Bollinger Bands.*
        Investopedia - Bollinger Bands
            https://www.investopedia.com/terms/b/bollingerbands.asp
    """
    mid = mu.sma(close, window)
    sd = mu.rolling_std(close, window)

    upper = mid + n_std * sd
    lower = mid - n_std * sd

    return pd.DataFrame(
        {
            "bb_mid": mid,
            "bb_upper": upper,
            "bb_lower": lower,
        },
        index=close.index,
    )


def returns(
    prices: pd.Series,
    *,
    horizons: Sequence[int] = (1, 5, 10, 21),
    kind: str = "both",
) -> pd.DataFrame:
    """Compute log or simple rate of returns of a price series.

    Parameters:
        prices (pd.Series): Series of prices.
        kind (str, optional): Type of returns to compute: "simple", "log", or "both".
            Default is "both".
        horizons (Sequence[int], optional): Sequence of time horizons (in periods) over
            which to compute returns. Default is (1, 5, 10, 21).

    Returns:
        pd.DataFrame: DataFrame containing the computed returns.

    Sources:
        Gregory Gundersen (06 February 2022). - *Returns and Log Returns.*
            https://gregorygundersen.com/blog/2022/02/06/log-returns/
        Investopedia - Rate of Return (ROR)
            https://www.investopedia.com/terms/r/rateofreturn.asp
    """
    prices = validate_prices(prices)

    if prices.empty:
        return pd.DataFrame(index=prices.index)

    if kind not in {"simple", "log", "both"}:
        raise ValueError("kind must be one of {'simple', 'log', 'both'}.")

    if not isinstance(horizons, list | tuple) or len(horizons) == 0:
        raise ValueError("horizons must be a non-empty sequence of positive integers.")

    # horizons validation
    hs: list[int] = []
    for h in horizons:
        if not isinstance(h, int):
            raise TypeError("Each horizon must be an integer.")
        if h <= 0:
            raise ValueError("Each horizon must be a positive integer.")
        hs.append(h)

    x = pd.to_numeric(prices, errors="coerce").astype(float)

    out: dict[str, pd.Series] = {}
    for h in hs:
        shifted = x.shift(h)

        if kind in {"simple", "both"}:
            # simple return: Pt / P{t-h} - 1
            r = x / shifted - 1.0
            out[f"ret_{h}"] = r

        if kind in {"log", "both"}:
            # log return: log(Pt/P{t-h}) = log(Pt) - log(P{t-h})
            lr = np.log(x) - np.log(shifted)
            out[f"logret_{h}"] = lr

    return pd.DataFrame(out, index=prices.index)


def volatility(
    prices: pd.Series,
    *,
    window: int | None = None,
    kind: str = "log",
    df: int = 0,
) -> float:
    """Volatility of price returns.

    The Volatility measures the dispersion of returns by computing the standard deviation
    of price over a period. Can be computed using log returns or simple returns.

    Parameters:
        prices (pd.Series): Series of prices.
        window (int, optional): Number of periods to use for calculating the volatility,
            by default 21.
        kind (str, optional): Calculation method for returns: "log" or "simple", by default "log".
        df (int, optional): Delta degrees of freedom to compute standard deviation, by default 0.
    """
    # Parameter validation
    prices = validate_prices(prices)

    if prices.empty:
        return float("nan")

    if kind not in {"log", "simple"}:
        raise ValueError("kind must be one of {'log', 'simple'}.")
    if window is not None and (not isinstance(window, int) or window <= 0):
        raise ValueError("window must be a positive integer or None.")
    if df not in (0, 1):
        raise ValueError("df must be 0 or 1.")

    x = pd.to_numeric(prices, errors="coerce").astype(float)

    if kind == "log":
        x = x.where(x > 0)
        rets = (np.log(x) - np.log(x.shift(1))).dropna()
    else:
        rets = x.pct_change().dropna()

    if window is not None:
        rets = rets.tail(window)

    if rets.size == 0:
        return float("nan")

    return float(np.std(rets, ddof=df))


def rolling_volatility(
    prices: pd.Series,
    *,
    window: int,
    kind: str = "log",  # "log" or "simple"
    df: int = 0,
) -> pd.Series:
    """Rolling Volatility of price returns over specified windows.

    The Rolling Volatility computes the volatility price movements over specified windows, giving
    insights into market volatility trends and risks. It can be computed using log returns or
    simple returns.

    Parameters:
        prices (pd.Series): Series of prices.
        window (int): Window size (in periods) for calculating the rolling volatility.
        kind (str, optional): Calculation method for returns: "log" or "simple", by default "log".
        df (int, optional): Delta degrees of freedom to compute standard deviation, by default 0.

    """
    # Parameter validation
    prices = validate_prices(prices)

    if prices.empty:
        return pd.Series(index=prices.index)

    if not isinstance(window, int) or window <= 0:
        raise ValueError("window must be a positive integer.")
    if kind not in {"log", "simple"}:
        raise ValueError("kind must be one of {'log', 'simple'}.")
    if df not in (0, 1):
        raise ValueError("df must be 0 or 1.")

    x = pd.to_numeric(prices, errors="coerce").astype(float)
    if kind == "log":
        x = x.where(x > 0)
        returns = np.log(x) - np.log(x.shift(1))
    else:
        returns = x.pct_change()

    return mu.rolling_std(
        returns,
        window=window,
        df=df,
        min_periods=window,
    )
