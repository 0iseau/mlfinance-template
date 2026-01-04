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
    lags: Annotated[bool, typer.Option("--lags")] = False,
) -> None:
    """Build features from data."""
    if not data_csv.exists():
        raise typer.BadParameter("CSV file not found")

    # Exclusivity
    selected = sum([analyze, summary, lags])
    if selected > 1:
        raise typer.BadParameter("Use only one of --analyze / --summary / --lags")
    option = "analyze"
    if summary:
        option = "summary"
    elif lags:
        option = "lags"

    data = pd.read_csv(data_csv)
    out = mf.build_feature(data, option)

    path = Path("out_features.csv")
    out.to_csv(path, index=False)
