"""Backtesting utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import mlfinance.features as mf


def _equity_curve(returns: pd.Series) -> pd.Series:
    r = returns.fillna(0.0).astype(float)
    return (1.0 + r).cumprod()


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def _sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna().astype(float)
    if r.empty:
        return float("nan")
    std = float(r.std(ddof=0))
    if std == 0.0:
        return float("nan")
    return float(r.mean() / std * np.sqrt(periods_per_year))


def _sortino_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna().astype(float)
    if r.empty:
        return float("nan")
    downside = r[r < 0.0]
    downside_std = float(downside.std(ddof=0))
    if downside_std == 0.0:
        return float("nan")
    return float(r.mean() / downside_std * np.sqrt(periods_per_year))


def _compute_metrics(returns: pd.Series, periods_per_year: int = 252) -> dict[str, float]:
    equity = _equity_curve(returns)
    return {
        "sharpe": _sharpe_ratio(returns, periods_per_year=periods_per_year),
        "sortino": _sortino_ratio(returns, periods_per_year=periods_per_year),
        "max_drawdown": _max_drawdown(equity),
    }


def _turnover_from_position(position: pd.Series) -> pd.Series:
    pos = position.astype(float)
    turnover = pos.diff().abs()
    if not pos.empty:
        turnover.iloc[0] = abs(float(pos.iloc[0]))
    return turnover.fillna(0.0)


def _prepare_xy_for_returns_prediction(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if "Returns_t+1" not in df.columns:
        raise ValueError("Missing target column: Returns_t+1")

    y = df["Returns_t+1"].astype(float)

    # Exclude targets; keep only numeric predictors
    target_cols = ["Returns_t+1", "Direction", "RVOL_t+1"]
    X = df.drop(columns=target_cols, errors="ignore")
    X = X.select_dtypes(include=["number"]).copy()

    # Keep rows where target exists
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    # Remove NaN, inf in features
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    y = y.loc[X.index]

    return X, y


def backtest_ml_pred(
    data: pd.DataFrame,
    *,
    cost: float = 0.0,
    train_frac: float = 0.6,
    step: int = 1,
    periods_per_year: int = 252,
    random_state: int = 0,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Run an expanding-window walk-forward backtest for a ML prediction strategy.

    Strategy:
            - Build features using `mlfinance.features.build_feature(..., option="analyze")`.
            - Predict next-period returns (Returns_t+1) with Ridge regression.
            - Long/flat position: 1 if prediction > 0 else 0.
            - Transaction costs: proportional to turnover: cost * |pos_t - pos_{t-1}|.

    Returns:
            backtest_df: per-period backtest outputs for the out-of-sample region.
            metrics: dict containing strategy and benchmark metrics.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if cost < 0.0:
        raise ValueError("cost must be >= 0")
    if not (0.0 < train_frac < 1.0):
        raise ValueError("train_frac must be in (0, 1)")
    if step <= 0:
        raise ValueError("step must be >= 1")
    if not isinstance(random_state, int):
        raise TypeError("random_state must be an integer")

    df = mf.build_feature(data, option="analyze", features="all")
    if df.empty:
        raise ValueError("No usable data for backtest")

    X, y_next = _prepare_xy_for_returns_prediction(df)
    n = len(X)
    if n < 10:
        raise ValueError("Not enough data after cleaning to run backtest")

    train_end = int(train_frac * n)
    train_end = max(train_end, 5)
    train_end = min(train_end, n - 1)

    pred = pd.Series(index=X.index, dtype=float)
    model = Ridge(alpha=1.0, fit_intercept=True)

    # Expanding-window walk-forward
    for start in range(train_end, n, step):
        end = min(start + step, n)
        X_train = X.iloc[:start]
        y_train = y_next.iloc[:start]
        X_test = X.iloc[start:end]

        if len(X_train) < 2 or X_test.empty:
            continue

        model.fit(X_train, y_train)
        pred.iloc[start:end] = model.predict(X_test)

    # Evaluate only on out-of-sample points where we have predictions
    oos_mask = pred.notna()
    if not bool(oos_mask.any()):
        raise ValueError("Backtest produced no out-of-sample predictions")

    pred_oos = pred.loc[oos_mask]
    y_oos = y_next.loc[oos_mask]

    position = (pred_oos > 0.0).astype(int)
    turnover = _turnover_from_position(position)

    strat_ret = position.astype(float) * y_oos - float(cost) * turnover
    bh_ret = y_oos.astype(float)

    strat_equity = _equity_curve(strat_ret)
    bh_equity = _equity_curve(bh_ret)

    out = pd.DataFrame(
        {
            "y_true": y_oos,
            "y_pred": pred_oos,
            "position": position,
            "turnover": turnover,
            "strategy_return": strat_ret,
            "buy_hold_return": bh_ret,
            "strategy_equity": strat_equity,
            "buy_hold_equity": bh_equity,
        },
        index=pred_oos.index,
    )

    # Helpful date column if present in original cleaned df
    if "Date" in df.columns:
        out.insert(0, "Date", df.loc[out.index, "Date"].values)

    strat_metrics = _compute_metrics(strat_ret, periods_per_year=periods_per_year)
    bh_metrics = _compute_metrics(bh_ret, periods_per_year=periods_per_year)

    metrics = {
        "strategy": strat_metrics,
        "buy_and_hold": bh_metrics,
    }

    return out, metrics


def backtest_ma_cross(
    data: pd.DataFrame,
    *,
    cost: float = 0.0,
    periods_per_year: int = 252,
    fast: int = 20,
    slow: int = 50,
    random_state: int = 0,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Backtest a classic moving-average crossover (trend-following) strategy.

    Strategy:
        - Compute fast/slow simple moving averages of Price.
        - Long/flat: 1 if SMA_fast > SMA_slow else 0.
        - Apply position to next-period returns (Returns_t+1).
        - Transaction costs: proportional to turnover: cost * |pos_t - pos_{t-1}|.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if cost < 0.0:
        raise ValueError("cost must be >= 0")
    if not isinstance(fast, int) or not isinstance(slow, int):
        raise TypeError("fast and slow must be integers")
    if fast <= 0 or slow <= 0:
        raise ValueError("fast and slow must be positive")
    if fast >= slow:
        raise ValueError("fast must be < slow")
    if not isinstance(random_state, int):
        raise TypeError("random_state must be an integer")

    # Seed any stochastic components used during backtesting
    np.random.seed(random_state)

    df = mf.build_feature(data, option="build")
    if df.empty:
        raise ValueError("No usable data for backtest")

    if "Price" not in df.columns or "Returns_t+1" not in df.columns:
        raise ValueError("Backtest requires Price and Returns_t+1 columns")

    price = df["Price"].astype(float)
    y_next = df["Returns_t+1"].astype(float)

    sma_fast = price.rolling(window=fast, min_periods=fast).mean()
    sma_slow = price.rolling(window=slow, min_periods=slow).mean()

    raw_position = (sma_fast > sma_slow).astype(int)
    mask = y_next.notna() & raw_position.notna()
    if not bool(mask.any()):
        raise ValueError("Not enough data after cleaning to run backtest")

    y_oos = y_next.loc[mask]
    position = raw_position.loc[mask]
    turnover = _turnover_from_position(position)

    strat_ret = position.astype(float) * y_oos - float(cost) * turnover
    bh_ret = y_oos.astype(float)

    strat_equity = _equity_curve(strat_ret)
    bh_equity = _equity_curve(bh_ret)

    out = pd.DataFrame(
        {
            "y_true": y_oos,
            "y_pred": np.nan,
            "position": position,
            "turnover": turnover,
            "strategy_return": strat_ret,
            "buy_hold_return": bh_ret,
            "strategy_equity": strat_equity,
            "buy_hold_equity": bh_equity,
        },
        index=y_oos.index,
    )

    if "Date" in df.columns:
        out.insert(0, "Date", df.loc[out.index, "Date"].values)

    strat_metrics = _compute_metrics(strat_ret, periods_per_year=periods_per_year)
    bh_metrics = _compute_metrics(bh_ret, periods_per_year=periods_per_year)

    metrics = {
        "strategy": strat_metrics,
        "buy_and_hold": bh_metrics,
    }

    return out, metrics
