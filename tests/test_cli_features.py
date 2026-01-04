from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from mlfinance.cli import app

runner = CliRunner()


def _write_sample_ohlcv_csv(path: Path, n: int = 80) -> Path:
    # Keep prices strictly positive to satisfy feature validation.
    close = pd.Series(range(100, 100 + n), dtype=float)
    df = pd.DataFrame(
        {
            "close": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "volume": pd.Series(range(1_000, 1_000 + n), dtype=float),
        }
    )
    df.to_csv(path, index=False)
    return path


def _first_stdout_line(result) -> str:
    # Typer's CliRunner returns captured output in result.stdout.
    return (result.stdout or "").splitlines()[0] if (result.stdout or "").splitlines() else ""


def _combined_output(result) -> str:
    # Some CLI errors are written to stderr, depending on Click/Typer versions.
    # `Result.output` is the safest unified view.
    return result.output or ""


def test_features_missing_csv_file_errors(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.csv"
    result = runner.invoke(app, ["features", str(missing)])

    # BadParameter -> usage error.
    assert result.exit_code != 0
    assert "CSV file not found" in _combined_output(result)


@pytest.mark.parametrize(
    "flags",
    [
        ["--analyze", "--summary"],
        ["--analyze", "--lags"],
        ["--summary", "--lags"],
        ["--analyze", "--summary", "--lags"],
    ],
)
def test_features_flags_are_mutually_exclusive(tmp_path: Path, flags: list[str]) -> None:
    csv_path = _write_sample_ohlcv_csv(tmp_path / "data.csv")
    result = runner.invoke(app, ["features", str(csv_path), *flags])

    assert result.exit_code != 0
    assert "Use only one of --analyze / --summary / --lags" in _combined_output(result)


def test_features_invalid_csv_missing_required_columns_errors(tmp_path: Path) -> None:
    # CSV that pandas can read, but doesn't provide a usable price column.
    df = pd.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
    csv_path = tmp_path / "bad.csv"
    df.to_csv(csv_path, index=False)

    result = runner.invoke(app, ["features", str(csv_path), "--analyze"])
    assert result.exit_code != 0
