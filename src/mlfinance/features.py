"""Feature engineering module.

Provides functions that compute technical indicators and price-volume transformations
used as inputs to machine-learning models.

Functions:
    Technical indicators:

        rsi: Relative Strength Index using Wilder smoothing.
        macd: MACD line, signal line, and histogram.
        bollinger_bands: Bollinger bands.
    Price-based features:

        returns: Simple and log rate of returns.
        volatility, rolling_volatility, atr: Volatility, rolling volatility, and
            Average True Range of returns.
        momentum_ts, macd_v: Time series momentum, MACD-V

    Volume-based features:
        obv: On-Balance Volume.
        volume_ma: Volume Moving Average.
        rvol : Relative Volume.

    Lag and rolling statistics:
        lags: Lagged values of a time series.
        rolling_indicators: Rolling mean, std, min, max of a time series.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

import mlfinance.math_utils as mu
from mlfinance.validation import validate_data


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
    prices = validate_data(prices)

    if not isinstance(window, int):
        raise TypeError("window must be an integer.")
    if window <= 0:
        raise ValueError("window must be a positive integer.")
    if prices.empty:
        return pd.Series(index=prices.index, dtype=float, name=f"RSI_{window}")

    if (prices.notna() & prices.le(0.0)).any():
        raise ValueError("prices must be strictly positive.")

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
    # parameter validation
    prices = validate_data(prices)

    for name, v in (("fast", fast), ("slow", slow), ("signal", signal)):
        if not isinstance(v, int) or v <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    if fast >= slow:
        raise ValueError("fast must be < slow.")

    if prices.empty:
        macd_line = pd.Series(index=prices.index, dtype=float, name=f"MACD_{fast}_{slow}")

        signal_line = pd.Series(
            index=prices.index, dtype=float, name=f"MACDsig_{fast}_{slow}_{signal}"
        )

        hist = pd.Series(index=prices.index, dtype=float, name=f"MACDhist_{fast}_{slow}_{signal}")

        return macd_line, signal_line, hist

    if (prices.notna() & prices.le(0.0)).any():
        raise ValueError("prices must be strictly positive.")

    ema_fast = mu.ema(prices, span=fast, min_periods=fast)
    ema_slow = mu.ema(prices, span=slow, min_periods=slow)

    macd_line = (ema_fast - ema_slow).astype(float)
    macd_line.name = f"MACD_{fast}_{slow}"

    signal_line = mu.ema(macd_line, span=signal, min_periods=signal)
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

    Returns:
        pd.DataFrame: DataFrame containing the Bollinger Bands features

    Sources:
        Bollinger, J. (2002). *Bollinger on Bollinger Bands.*
        Investopedia - Bollinger Bands
            https://www.investopedia.com/terms/b/bollingerbands.asp
    """
    # Parameter validation
    close = validate_data(close)

    if not isinstance(window, int):
        raise TypeError("window must be an integer.")
    if window <= 0:
        raise ValueError("window must be a positive integer.")
    if not isinstance(n_std, (int | float)):
        raise TypeError("n_std must be a number.")
    if n_std < 0.0:
        raise ValueError("n_std must be positive.")
    if close.empty:
        return pd.DataFrame(index=close.index, columns=["bb_mid", "bb_upper", "bb_lower"])

    if (close.notna() & close.le(0.0)).any():
        raise ValueError("close prices must be strictly positive.")

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


def _validate_kind(kind: str) -> None:
    if kind not in {"simple", "log", "both"}:
        raise ValueError("kind must be one of {'simple', 'log', 'both'}.")


def returns(
    prices: pd.Series,
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
    prices = validate_data(prices)

    _validate_kind(kind)

    if not isinstance(horizons, list | tuple) or len(horizons) == 0:
        raise ValueError("horizons must be a non-empty sequence of positive integers.")

    if (prices.notna() & prices.le(0.0)).any():
        raise ValueError("prices must be strictly positive.")

    # horizons validation
    hs: list[int] = []
    for h in horizons:
        if not isinstance(h, int):
            raise TypeError("Each horizon must be an integer.")
        if h <= 0:
            raise ValueError("Each horizon must be a positive integer.")
        hs.append(h)

    if prices.empty:
        if kind == "both":
            cols = [f"ret_{h}" for h in hs] + [f"logret_{h}" for h in hs]
        elif kind == "simple":
            cols = [f"ret_{h}" for h in hs]
        else:
            cols = [f"logret_{h}" for h in hs]
        return pd.DataFrame(index=prices.index, columns=cols, dtype=float)

    x = prices

    out: dict[str, pd.Series] = {}

    for h in hs:
        shifted = x.shift(h)

        if kind in {"simple", "both"}:
            # simple return: Pt / P{t-h} - 1
            denom = shifted.where((shifted != 0.0) & shifted.notna())
            r = (x / denom - 1.0).astype(float)
            out[f"ret_{h}"] = r

        if kind in {"log", "both"}:
            # log return: log(Pt/P{t-h}) = log(Pt) - log(P{t-h})
            valid = x.gt(0.0) & shifted.gt(0.0)
            lr = (np.log(x) - np.log(shifted)).where(valid).astype(float)
            out[f"logret_{h}"] = lr

    return pd.DataFrame(out, index=prices.index)


def volatility(
    prices: pd.Series,
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
            by default None (uses all available data).
        kind (str, optional): Calculation method for returns: "log" or "simple", by default "log".
        df (int, optional): Delta degrees of freedom to compute standard deviation, by default 0.

    Returns:
        (float): Volatility of the price returns.

    Sources:
        Investopedia - Volatility
            https://www.investopedia.com/terms/v/volatility.asp
    """
    # Parameter validation
    prices = validate_data(prices)

    if prices.empty:
        return float("nan")

    if kind not in {"log", "simple"}:
        raise ValueError("kind must be one of {'log', 'simple'}.")
    if window is not None and (not isinstance(window, int) or window <= 0):
        raise ValueError("window must be a positive integer or None.")
    if df not in (0, 1):
        raise ValueError("df must be 0 or 1.")

    if (prices.notna() & prices.le(0.0)).any():
        raise ValueError("prices must be strictly positive.")

    x = prices.astype(float)

    if kind == "log":
        # Guard against non-positive values (will become NaN instead of -inf / warnings)
        x_pos = x.where(x > 0.0)
        log_x_pos = pd.Series(
            np.log(x_pos.to_numpy(dtype=float, na_value=np.nan)),
            index=x_pos.index,
            dtype=float,
        )
        rets = log_x_pos.diff()
    else:
        rets: pd.Series[float] = x.pct_change()  # type: ignore

    rets = rets.replace([np.inf, -np.inf], np.nan).dropna()

    if window is not None:
        rets = rets.tail(window)

    if rets.size == 0:
        return 0.0

    return float(np.std(rets.to_numpy(dtype=float), ddof=df))


def rolling_volatility(
    prices: pd.Series,
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

    Returns:
        (pd.Series): Series containing the rolling volatility values.

    Sources:
        TradingView - Rolling Volatility Indicator
            https://www.tradingview.com/script/RCRc38L9-Rolling-Volatility-Indicator/


    """
    # Parameter validation
    prices = validate_data(prices)

    if prices.empty:
        return pd.Series(index=prices.index, dtype=float)

    if not isinstance(window, int) or window <= 0:
        raise ValueError("window must be a positive integer.")
    if kind not in {"log", "simple"}:
        raise ValueError("kind must be one of {'log', 'simple'}.")
    if df not in (0, 1):
        raise ValueError("df must be 0 or 1.")

    if (prices.notna() & prices.le(0.0)).any():
        raise ValueError("prices must be strictly positive.")

    x = prices

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


def momentum_ts(prices: pd.Series, window: int = 250) -> pd.DataFrame:
    """Time series momentum of price returns.

    Time series momentum, or absolute momentum, predicts an asset's future performance based
    on its passed trends over a given time frame. Can be plotted againts a 0% horizontal line
    to identify positive or negative momentum.

    parameters:
        prices (pd.Series): Series of prices.
        window (int, optional): Number of periods to use for calculating the momentum,
            by default 250.

    Returns:
        (pd.Series): Series containing the time series momentum values.

    Sources:
        Moskowitz, Tobias J., Yao Hua Ooi, and Lasse Heje Pedersen (2012). *Time Series Momentum.*
            Journal of Financial Economics 104 (2): 228-250.
        TradingView - Time Series Momentum (Absolute Momentum)
            https://de.tradingview.com/script/1qEJO8Wo/
    """
    prices = validate_data(prices)

    if not isinstance(window, int):
        raise TypeError("window must be an integer.")
    if window <= 0:
        raise ValueError("window must be a positive integer.")

    if (prices.notna() & prices.le(0.0)).any():
        raise ValueError("prices must be strictly positive.")

    if prices.empty:
        return pd.DataFrame(index=prices.index, columns=[f"mom_{window}"])

    x = prices

    shifted = x.shift(window)
    mom = x / shifted - 1.0

    return pd.DataFrame({f"mom_{window}": mom}, index=prices.index)


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """Average True Range (ATR) Volatility.

    The Average True Range (ATR) measures market volatility decomposing daily price ranges,
    including overnight gaps, permitting true movement assessment and not only intraday changes.

    Parameters:
        high (pd.Series): Series of high prices.
        low (pd.Series): Series of low prices.
        close (pd.Series): Series of closing prices.
        window (int, optional): Number of periods to use for calculating the ATR, by default 14.

    Returns:
        (pd.Series): Series containing the ATR values.

    Source:
        Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*
        Investopedia - Average True Range (ATR)
            https://www.investopedia.com/terms/a/atr.asp
    """
    # Parameter validation
    high = validate_data(high)
    low = validate_data(low)
    close = validate_data(close)

    if high.empty or low.empty or close.empty:
        return pd.Series(index=close.index, dtype=float, name=f"ATR_{window}")

    if not isinstance(window, int):
        raise TypeError("window must be an integer.")
    if window <= 0:
        raise ValueError("window must be a positive integer.")

    if not (high.index.equals(low.index) and high.index.equals(close.index)):
        raise ValueError("high, low, close must have the same index.")

    # Validate strictly positive prices
    for series, name in ((high, "high"), (low, "low"), (close, "close")):
        if (series.notna() & series.le(0.0)).any():
            raise ValueError(f"{name} prices must be strictly positive.")

    # Prepare True Range (TR)
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Wilder ATR
    atr = pd.Series(np.nan, index=tr.index, dtype=float)

    tr_valid = tr.dropna()
    if tr_valid.size < window:
        atr.name = f"ATR_{window}"
        return atr

    # Initial ATR (SMA of first window TR values)
    seed_idx = tr_valid.index[window - 1]

    # Force an integer positional location for mypy
    seed_loc_arr = tr.index.get_indexer(pd.Index([seed_idx]))
    seed_loc = int(seed_loc_arr[0])
    if seed_loc < 0:
        atr.name = f"ATR_{window}"
        return atr

    atr.iloc[seed_loc] = tr_valid.iloc[:window].mean()

    # Wilder smoothing
    for i in range(seed_loc + 1, len(tr)):
        if np.isnan(tr.iloc[i]):
            atr.iloc[i] = np.nan
        else:
            atr.iloc[i] = (atr.iloc[i - 1] * (window - 1) + tr.iloc[i]) / window

    atr.name = f"ATR_{window}"

    return atr


def macd_v(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    fast: int = 12,
    slow: int = 26,
    atr_window: int = 26,
) -> pd.Series:
    """MACD-V Volatility Normalized Momentum.

    MACD-V uses MACD normalizing by volatility itself. The tool of measurement of volatility
    used by the author is the Average True Range (ATR), but the simple and rolling volatility
    can be used as well.

    Computation:
    MACD-V = (EMA_fast(close) - EMA_slow(close)) / ATR(atr_length)

    Parameters:
        close (pd.Series): Series of closing prices.
        high (pd.Series): Series of high prices.
        low (pd.Series): Series of low prices.
        fast (int, optional): Span for the fast EMA, by default 12.
        slow (int, optional): Span for the slow EMA, by default 26.
        atr_window (int, optional): Window for the ATR calculation, by default 26.

    Returns:
        (pd.Series): Series containing the MACD-V values.

    Sources:
        Alex Spiroglou (3 May 2022). - *MACD-V: Volatility Normalized Momentum.*
        TradingView - MACD-V Volatility Normalized Momentum
            https://www.tradingview.com/script/CE4EqZ1A-MACD-V-Volatility-Normalized-Momentum/
    """
    # Parameter validation
    close = validate_data(close)
    high = validate_data(high)
    low = validate_data(low)

    if close.empty or high.empty or low.empty:
        return pd.Series(index=close.index, dtype=float, name="macd_v")

    if not all(isinstance(x, int) and x > 0 for x in (fast, slow, atr_window)):
        raise ValueError("fast, slow, and atr_window must be positive integers.")

    if not (close.index.equals(high.index) and close.index.equals(low.index)):
        raise ValueError("high, low, close must have the same index.")

    # Validate strictly positive prices
    for series, name in ((close, "close"), (high, "high"), (low, "low")):
        if (series.notna() & series.le(0.0)).any():
            raise ValueError(f"{name} prices must be strictly positive.")

    # MACD
    ema_fast = mu.ema(close, span=fast, min_periods=fast)
    ema_slow = mu.ema(close, span=slow, min_periods=slow)
    macd = (ema_fast - ema_slow).astype(float)

    # ATR normalization (Wilder)
    atr_val = atr(high, low, close, window=atr_window).astype(float)

    # Avoid division by zero / missing ATR
    denom = atr_val.where((atr_val != 0.0) & atr_val.notna())
    macd_v = (macd / denom).astype(float)
    macd_v.name = "macd_v"

    return macd_v


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume (OBV).

    The OBV measures the relative buying/selling pressures on an asset using the flows
    of traded volume to predict the future short-term price changes.

    Parameters:
        close (pd.Series): Series of closing prices.
        volume (pd.Series): Series of traded volumes.

    Returns:
        (pd.Series): Series containing the OBV values.

    Source:
        Joseph E. Granville (1976). - *New Strategy of Daily Stock Market Timing for Maximum
            Profit*
        Corporate Finance Institute - On-Balance Volume indicator (OBV).
    """
    # parameter validation
    close = validate_data(close)
    volume = validate_data(volume)

    if close.empty or volume.empty:
        return pd.Series(index=close.index, dtype=float, name="obv")

    if not close.index.equals(volume.index):
        raise ValueError("close and volume must have the same index.")

    c = close
    v = volume

    if (c.notna() & c.le(0.0)).any():
        raise ValueError("close prices must be strictly positive.")
    if (v.notna() & v.lt(0.0)).any():
        raise ValueError("volume must be non-negative.")

    d = c.diff()

    direction = pd.Series(
        np.sign(d.to_numpy(dtype=float, na_value=np.nan)),
        index=d.index,
        dtype=float,
    ).fillna(0.0)

    signed_vol = (v * direction).astype(float)
    signed_vol[(c.isna()) | (v.isna())] = np.nan

    out = signed_vol.fillna(0.0).cumsum()
    out[signed_vol.isna()] = np.nan

    out.name = "obv"
    return out


def volume_ma(volume: pd.Series, window: int) -> pd.Series:
    """Volume Moving Average (VMA).

    The VMA is the average of the volume of an asset over a specified window.
    It helps explain volume trends by smoothing out short-term fluctuations.

    Parameters:
        volume (pd.Series): Series of traded volumes.
        window (int): Number of periods to use for calculating the VMA.

    Returns:
        (pd.Series): Series containing the VMA values.

    Source:
        Wikipedia - Moving average
            https://en.wikipedia.org/wiki/Moving_average
    """
    volume = validate_data(volume)

    if not isinstance(window, int) or window <= 0:
        raise ValueError("window must be a positive integer.")
    if volume.empty:
        return pd.Series(index=volume.index, dtype=float, name=f"VMA_{window}")

    if (volume.notna() & volume.lt(0.0)).any():
        raise ValueError("volume must be non-negative.")

    vma = mu.sma(volume, window)

    vma.name = f"VMA_{window}"

    return vma


def rvol(volume: pd.Series, window: int) -> pd.Series:
    """Relative Volume (RVOL).

    The RVOL compares the current volume to its rolling volatility over a specified
    window.

    Parameters:
        volume (pd.Series): Series of traded volumes.
        window (int): Number of periods to use for calculating the RVOL.

    Returns:
        (pd.Series): Series containing the RVOL values.

    Source:
        dastrader - RVOL - Relative Volume
            https://dastrader.com/docs/rvol-relative-volume/
    """
    volume = validate_data(volume)

    if volume.empty:
        return pd.Series(index=volume.index, dtype=float, name="rvol")

    if not isinstance(window, int) or window <= 0:
        raise ValueError("window must be a positive integer.")

    v = volume
    if (v.notna() & v.lt(0.0)).any():
        raise ValueError("volume must be non-negative.")

    sma_vol = mu.sma(v, window, min_periods=window).astype(float)
    denom = sma_vol.where((sma_vol != 0.0) & sma_vol.notna())

    rvol = (v / denom).astype(float)
    rvol.name = "rvol"

    return rvol


def lags(data: pd.Series, shift: int = 1) -> pd.DataFrame:
    """Lagged values.

    Shift in time series values, used to analyze past dependencies.
    Permits the program to compute indicators from the past without
    using forward data.

    Parameters:
        data (pd.Series): Time series data.
        shift (int): Steps before current observation. By default, 1.

    Return:
        (pd.DataFrame): Data frame containing lagged values.

    Source:
        WikiPedia - Lag operator
            https://en.wikipedia.org/wiki/Lag_operator
    """
    data = validate_data(data)
    # No range validity as method is for general use.

    if not isinstance(shift, int) or shift <= 0:
        raise ValueError("shift must be a positive integer.")

    if data.empty:
        return pd.DataFrame(
            index=data.index, dtype=float, columns=[f"lag_{k}" for k in range(0, shift + 1)]
        )

    return pd.DataFrame({f"lag_{k}": data.shift(k) for k in range(0, shift + 1)}, index=data.index)


def rolling_indicators(data: pd.Series, window: int) -> pd.DataFrame:
    """Rolling mean, standard deviation, minimum, maximum.

    Computes rolling indicators of a series over a specified window.

    Parameters:
        data (pd.Series): Data series
        window (int): Time period to use for calculation.

    Returns:
        (pd.DataFrame): Data frame with rolling indicators.
    """
    data = validate_data(data)
    # No range validity as method is for general use.

    if not isinstance(window, int) or window <= 0:
        raise ValueError("window must be a positive integer.")

    if data.empty:
        return pd.DataFrame(
            index=data.index,
            dtype=float,
            columns=[
                f"roll_mean_{window}",
                f"roll_std_{window}",
                f"roll_min_{window}",
                f"roll_max_{window}",
            ],
        )

    r = data.rolling(window=window, min_periods=window)
    return pd.DataFrame(
        {
            f"roll_mean_{window}": r.mean().astype(float),
            f"roll_std_{window}": mu.rolling_std(x=data, window=window).astype(float),
            f"roll_min_{window}": r.min().astype(float),
            f"roll_max_{window}": r.max().astype(float),
        },
        index=data.index,
    )


def _build_analyze(x: pd.DataFrame, features: str = "all") -> pd.DataFrame:
    out = pd.DataFrame(index=x.index, dtype=float)

    if "Date" in x.columns:
        out["Date"] = x["Date"]
    out["Price"] = x["Price"]

    if features in {"all", "technical"}:
        out["rsi"] = rsi(x["Price"])
        out["macd_line"], out["signal_line"], out["macd_hist"] = macd(x["Price"])
        out["bb_mid"], out["bb_upper"], out["bb_lower"] = bollinger_bands(x["Price"]).T.values
        out["momentum_ts"] = momentum_ts(x["Price"])["mom_250"]

    if features in {"all", "stat"}:
        out["returns"] = returns(x["Price"])["ret_1"]
        out["volatility"] = volatility(x["Price"])
        out["rolling_volatility"] = rolling_volatility(x["Price"], window=21)
        out["roll_mean_10"] = rolling_indicators(x["Price"], window=10)["roll_mean_10"]
        out["roll_std_10"] = rolling_indicators(x["Price"], window=10)["roll_std_10"]
        out["roll_min_10"] = rolling_indicators(x["Price"], window=10)["roll_min_10"]
        out["roll_max_10"] = rolling_indicators(x["Price"], window=10)["roll_max_10"]

    if "Volume" in x.columns:
        out["Volume"] = x["Volume"]
        if features in {"all", "technical"}:
            out["volume_ma"] = volume_ma(x["Volume"], window=20)
            out["rvol"] = rvol(x["Volume"], window=20)
            out["obv"] = obv(x["Price"], x["Volume"])
        if features in ("all", "stat"):
            out["vol_roll_mean_10"] = rolling_indicators(x["Volume"], window=10)["roll_mean_10"]
            out["vol_roll_std_10"] = rolling_indicators(x["Volume"], window=10)["roll_std_10"]
            out["vol_roll_min_10"] = rolling_indicators(x["Volume"], window=10)["roll_min_10"]
            out["vol_roll_max_10"] = rolling_indicators(x["Volume"], window=10)["roll_max_10"]

    required_cols = {"High", "Low", "Price"}
    has_required_cols = required_cols.issubset(x.columns)
    is_technical = features in {"all", "technical"}

    if has_required_cols and is_technical:
        out["atr"] = atr(x["High"], x["Low"], x["Price"])
        out["macd_v"] = macd_v(x["Price"], x["High"], x["Low"])

    out["Returns_t+1"] = x["Returns_t+1"]
    out["Direction"] = x["Direction"]
    if "RVOL_t+1" in x.columns:
        out["RVOL_t+1"] = x["RVOL_t+1"]

    return out


def _build_summary(x: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(dtype=float)

    out["number of observations"] = [len(x)]
    if "Date" in x.columns:
        out["date range"] = [f"{x['Date'].min()} to {x['Date'].max()}"]
    out["number of missing prices"] = [x["Price"].isna().sum()]
    out["price_mean"] = [x["Price"].mean()]
    out["price_std"] = [x["Price"].std()]
    out["price_var"] = [x["Price"].var()]
    out["min_price"] = [x["Price"].min()]
    out["max_price"] = [x["Price"].max()]
    out["median_price"] = [x["Price"].median()]
    if "Returns_t+1" in x.columns:
        out["number of missing returns"] = [x["Returns_t+1"].isna().sum()]
        out["returns_mean"] = [x["Returns_t+1"].mean()]
        out["returns_std"] = [x["Returns_t+1"].std()]
        out["returns_var"] = [x["Returns_t+1"].var()]
        out["min_returns"] = [x["Returns_t+1"].min()]
        out["max_returns"] = [x["Returns_t+1"].max()]
        out["median_returns"] = [x["Returns_t+1"].median()]
    # Rolling indicators
    for w in (10, 21):
        ri = rolling_indicators(x["Price"], window=w)
        out[f"roll_mean_{w}"] = [ri[f"roll_mean_{w}"].mean()]
        out[f"roll_std_{w}"] = [ri[f"roll_std_{w}"].mean()]
        out[f"roll_min_{w}"] = [ri[f"roll_min_{w}"].mean()]
        out[f"roll_max_{w}"] = [ri[f"roll_max_{w}"].mean()]

    if "Volume" in x.columns:
        out["number of missing volumes"] = [x["Volume"].isna().sum()]
        out["volume_mean"] = [x["Volume"].mean()]
        out["volume_std"] = [x["Volume"].std()]
        out["volume_var"] = [x["Volume"].var()]
        out["min_volume"] = [x["Volume"].min()]
        out["max_volume"] = [x["Volume"].max()]
        # Rolling indicators
        for w in (10, 21):
            vi = rolling_indicators(x["Volume"], window=w)
            out[f"vol_roll_mean_{w}"] = [vi[f"roll_mean_{w}"].mean()]
            out[f"vol_roll_std_{w}"] = [vi[f"roll_std_{w}"].mean()]
            out[f"vol_roll_min_{w}"] = [vi[f"roll_min_{w}"].mean()]
            out[f"vol_roll_max_{w}"] = [vi[f"roll_max_{w}"].mean()]
    return out


def _build_clean(x: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(x, pd.DataFrame):
        try:
            x = pd.DataFrame(x)
        except Exception as e:
            raise TypeError("Could not convert x to a pandas DataFrame.") from e

    if x.empty:
        return pd.DataFrame(index=x.index, dtype=float)

    # BY PRIORITY ORDER
    usable_prices = ("Price", "price", "PRICE", "Close", "close", "CLOSE", "Open", "Open", "OPEN")
    usable_highs = ("High", "high", "HIGH", "H", "h")
    usable_lows = ("Low", "low", "LOW", "L", "l")
    usable_volumes = ("Vol", "vol", "VOL", "Volume", "volume", "VOLUME")

    df = pd.DataFrame(index=x.index, dtype=float)

    usable_dates = (
        "Date",
        "date",
        "DATE",
        "Datetime",
        "datetime",
        "DATETIME",
        "Time",
        "time",
        "TIME",
    )

    temp_date = next((i for i in x.columns if i in usable_dates), None)
    temp_high = next((i for i in x.columns if i in usable_highs), None)
    temp_low = next((i for i in x.columns if i in usable_lows), None)
    temp_volume = next((i for i in x.columns if i in usable_volumes), None)

    if temp_date is not None:
        df["Date"] = x[temp_date]

    df["Price"] = x[next(i for i in x.columns if i in usable_prices)]
    if "Price" not in df.columns:
        raise ValueError("Cannot find usable price column in DataFrame.")
    if temp_high is not None:
        df["High"] = x[temp_high]
    if temp_low is not None:
        df["Low"] = x[temp_low]
    if temp_volume is not None:
        df["Volume"] = x[temp_volume]
    df["Returns_t+1"] = df["Price"].pct_change().shift(-1)
    print("Returns are calculated using Price column and shifted by -1 for prediction.")
    df["Direction"] = (df["Returns_t+1"] > 0).astype(int)
    if "Volume" in df.columns:
        df["RVOL_t+1"] = rvol(df["Volume"], window=20).shift(-1)

    return df.sort_index()


def build_feature(
    data: pd.DataFrame,
    option: str = "analyze",
    features: str = "all",
) -> pd.DataFrame:
    """Build and store all indicators values in a DataFrame.

    Parameters:
        data (pd.DataFrame): DataFrame containing financial time series data.
            Expected columns: 'close' (or 'price' or 'open'), 'high', 'low', 'volume'.
        option (str): option of features building : analyze, summary or build.

    Returns:
        pd.DataFrame: DataFrame containing computed features.
    """
    # Parameter validation

    if not isinstance(option, str) or option not in ("analyze", "summary", "build"):
        raise TypeError("option must be 'analyze', 'summary' or 'build'")

    x = _build_clean(data)

    if x.empty:
        return pd.DataFrame(index=x.index, dtype=float)

    if option == "analyze":
        return _build_analyze(x, features=features)

    if option == "summary":
        return _build_summary(x)

    if option == "build":
        return x

    return pd.DataFrame(index=x.index, dtype=float)
