"""CLI tests for feature commands."""

from __future__ import annotations

import pandas as pd
from typer.testing import CliRunner

from mlfinance.cli import app


def test_cli_features_analyze(tmp_path) -> None:
    """`mlfinance features <csv> --analyze` runs and prints a summary."""
    df = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0, 106.0],
        }
    )
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    runner = CliRunner()
    result = runner.invoke(app, ["features", str(csv_path), "--analyze"])

    assert result.exit_code == 0
    assert "Computed" in result.stdout
    assert "Summary stats:" in result.stdout


def test_cli_features_preview(tmp_path) -> None:
    """Default mode prints a small preview instead of full CSV."""
    df = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]})
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    runner = CliRunner()
    result = runner.invoke(app, ["features", str(csv_path)])

    assert result.exit_code == 0
    # Should include at least one known feature column name.
    assert "RSI_14" in result.stdout


def test_cli_features_missing_close_column(tmp_path) -> None:
    """A helpful error is shown when required columns are missing."""
    df = pd.DataFrame({"price_x": [1.0, 2.0, 3.0]})
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    runner = CliRunner()
    result = runner.invoke(app, ["features", str(csv_path), "--analyze"])

    assert result.exit_code != 0
    assert "must contain a `close`" in result.output


def test_cli_features_parses_date_column(tmp_path) -> None:
    """If a date column exists, it is parsed and does not break feature computation."""
    df = pd.DataFrame(
        {
            "date": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-06",
            ],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        }
    )
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    runner = CliRunner()
    result = runner.invoke(app, ["features", str(csv_path), "--analyze"])

    assert result.exit_code == 0
    assert "Computed" in result.stdout
