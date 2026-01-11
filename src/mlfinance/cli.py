"""CLI module for mlfinance package.

Usage :
mlfinance
features <data.csv>
-- analyze : returns available indicators values
-- summary : returns summary statistics of indicators
-- lag : creates lagged features for time series data

train <data.csv>
-- target : defines target value to predict
returns
direction
volatility
-- model : defines prediction method
rf (Random Forest)
gb (Gradient Boosting)
-- features : defines features category for model training
technical
price
volume
rolling
"""

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

import mlfinance.backtest as mb
import mlfinance.features as mf
import mlfinance.models as mm

app = typer.Typer(no_args_is_help=True)


@app.callback()  # type: ignore[misc]
def main() -> None:
    """MLFinance command-line interface."""
    return


@app.command("features")  # type: ignore[misc]
def features_cmd(
    data_csv: Annotated[Path, typer.Argument()],
    analyze: Annotated[bool, typer.Option("--analyze")] = False,
    summary: Annotated[bool, typer.Option("--summary")] = False,
    build: Annotated[bool, typer.Option("--build")] = False,
) -> None:
    """Build features from data."""
    if not data_csv.exists():
        raise typer.BadParameter("CSV file not found")

    # Exclusivity
    selected = sum([analyze, summary, build])
    if selected > 1:
        raise typer.BadParameter("Use only one of --analyze / --summary / --build")
    option = "analyze"
    if summary:
        option = "summary"
    elif build:
        option = "build"

    data = pd.read_csv(data_csv)
    out = mf.build_feature(data, option)

    path = Path("results/out_features.csv")
    path.parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(path, index=False)
    print(f"Features CSV file created at: {path.resolve()}\n")


@app.command("train")  # type: ignore[misc]
def train(
    data_csv: Annotated[Path, typer.Argument()],
    target: Annotated[str, typer.Option("--target")] = "returns",
    model: Annotated[str, typer.Option("--model")] = "ridge",
    features: Annotated[str, typer.Option("--features")] = "all",
    shap: Annotated[bool, typer.Option("--shap")] = False,
) -> None:
    """Train regression models."""
    if not data_csv.exists():
        raise typer.BadParameter("CSV file not found")

    # Validate options
    if target not in ("returns", "direction", "rvol"):
        raise ValueError("target must be 'returns', 'direction' or 'rvol'.")
    if model not in ("ridge", "rf", "gb"):
        raise ValueError("model must be 'ridge', 'rf' or 'gb'.")
    if features not in ("technical", "stat", "all"):
        raise ValueError("features must be 'technical', 'stat' or 'all'.")

    data = pd.read_csv(data_csv)

    df = mf.build_feature(data, features=features, option="analyze")
    features_path = Path("results/out_features.csv")
    features_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(features_path, index=False)
    print("Features file created at:", features_path.resolve())

    if model == "ridge":
        r = mm.Ridge_model(df, alpha=1.0, target=target, features=features, shap=shap)

    elif model == "rf":
        r = mm.Random_Forest_Model(df, target=target, features=features, shap=shap)

    else:
        r = mm.gradient_boosting_model(df, target=target, features=features, shap=shap)

    result, importance, summary, shap_val = r

    # Print summary
    print("-" * 40 + "\n")
    print("Model training summary:")
    for k in summary:
        v = summary[k]
        if isinstance(v, float):
            print(f"  {k}: {v:.6g}")
        else:
            print(f"  {k}: {v}")

    # File outputs
    path = Path("results/out_model.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    print("Model results file created at:", path.resolve())

    path = Path("results/out_importance.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(path, index=False)
    print("Feature importance file created at:", path.resolve())

    if shap:
        shap_path = Path("results/out_shap_values.csv")
        shap_path.parent.mkdir(parents=True, exist_ok=True)
        if shap_val is not None:
            shap_val.to_csv(shap_path, index=False)
            print("SHAP values file created at:", shap_path.resolve())


@app.command("backtest")  # type: ignore[misc]
def backtest(
    data_csv: Annotated[Path, typer.Argument()],
    strategy: Annotated[str, typer.Option("--strategy")] = "ml-pred",
    cost: Annotated[float, typer.Option("--cost")] = 0.0,
) -> None:
    """Backtest a strategy on historical data."""
    if not data_csv.exists():
        raise typer.BadParameter("CSV file not found")
    if cost < 0.0:
        raise ValueError("cost must be >= 0")
    if strategy not in ("ml-pred", "ma-cross"):
        raise ValueError("strategy must be 'ml-pred' or 'ma-cross'.")

    data = pd.read_csv(data_csv)

    if strategy == "ml-pred":
        bt_df, metrics = mb.backtest_ml_pred(data, cost=cost)
    else:
        bt_df, metrics = mb.backtest_ma_cross(data, cost=cost)

    path = Path("results/out_backtest.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    bt_df.to_csv(path, index=False)
    print("Backtest results file created at:", path.resolve())

    print("-" * 40 + "\n")
    print("Backtest summary:")
    for section in ("strategy", "buy_and_hold"):
        print(f"{section}:")
        for k in ("sharpe", "sortino", "max_drawdown"):
            v = metrics.get(section, {}).get(k)
            if isinstance(v, float):
                print(f"  {k}: {v:.6g}")
            else:
                print(f"  {k}: {v}")
