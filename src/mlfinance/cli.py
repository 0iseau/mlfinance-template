"""Command-line interface for MLFinance.

This module provides the `mlfinance` CLI entrypoint configured in `pyproject.toml`.
Currently implemented commands:

- `mlfinance features <csv> --analyze`: Compute a minimal technical feature set and
  print a short analysis summary.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from mlfinance import features as feat

app = typer.Typer(add_completion=False, no_args_is_help=True)

CSV_PATH_ARG = typer.Argument(
    ..., exists=True, dir_okay=False, readable=True, help="Path to input CSV file."
)
ANALYZE_OPT = typer.Option(
    False, "--analyze", help="Print a summary analysis of computed features."
)


@app.callback()  # type: ignore[misc]
def main() -> None:
    """MLFinance command-line interface."""


@app.command("features")  # type: ignore[misc]
def features_cmd(
    csv_path: Path = CSV_PATH_ARG,
    analyze: bool = ANALYZE_OPT,
) -> None:
    """Compute a minimal technical feature set from a CSV file.

    The CSV must contain a `close` column (case-insensitive). If a `date`, `datetime`,
    or `timestamp` column exists, it will be parsed and used as the index.

    When `--analyze` is provided, prints missingness and basic summary stats.
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        raise typer.BadParameter("CSV file is empty.")

    df = df.rename(columns=lambda c: str(c).strip().lower())

    for time_col in ("date", "datetime", "timestamp"):
        if time_col in df.columns:
            parsed = pd.to_datetime(df[time_col], errors="coerce", utc=True)
            if parsed.notna().any():
                df = df.drop(columns=[time_col])
                df.index = pd.DatetimeIndex(parsed)
            break

    close_col = None
    for candidate in ("close", "adj_close", "adjclose", "price"):
        if candidate in df.columns:
            close_col = candidate
            break

    if close_col is None:
        raise typer.BadParameter("CSV must contain a `close` column (case-insensitive).")

    close = df[close_col].astype(float)

    rsi = feat.rsi(close, window=14)
    macd_line, macd_sig, macd_hist = feat.macd(close)
    bb = feat.bollinger_bands(close)

    out = pd.concat([rsi, macd_line, macd_sig, macd_hist, bb], axis=1)

    if analyze:
        n_rows = int(out.shape[0])
        n_cols = int(out.shape[1])

        typer.echo(f"Computed {n_cols} features for {n_rows} rows.")

        na_frac = out.isna().mean().sort_values(ascending=False)
        na_summary = na_frac[na_frac.gt(0.0)]
        if na_summary.empty:
            typer.echo("Missing values: none")
        else:
            typer.echo("Missing values (fraction):")
            typer.echo(na_summary.to_string(float_format=lambda x: f"{x:.3f}"))

        desc = out.describe().T
        cols = [c for c in ("mean", "std", "min", "max") if c in desc.columns]
        typer.echo("Summary stats:")
        typer.echo(desc[cols].to_string(float_format=lambda x: f"{x:.6g}"))
        return

    # Default: print a small preview to avoid dumping huge CSVs to stdout.
    typer.echo(out.head(5).to_string())


if __name__ == "__main__":  # pragma: no cover
    app()
