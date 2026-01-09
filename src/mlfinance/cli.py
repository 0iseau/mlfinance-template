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

    path = Path("out_features.csv")

    out.to_csv(path, index=False)
    print(f"CSV file created at: {path.resolve()}\n")
    print(f"Columns generated : {out.columns.tolist()}")


@app.command("train")  # type: ignore[misc]
def train(
    data_csv: Annotated[Path, typer.Argument()],
    target: Annotated[str, typer.Option("--target")] = "returns",
    model: Annotated[str, typer.Option("--model")] = "ridge",
    features: Annotated[str, typer.Option("--features")] = "all",
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
    print(f"Successfully built features for: {df.columns.tolist()}")

    if model == "ridge":
        r = mm.Ridge_model(df, alpha=1.0, target=target)

    result, summary = r

    # Print summary
    print("Model training summary:")
    print(summary)

    # File outputs
    path = Path("out_model.csv")
    result.to_csv(path, index=False)
