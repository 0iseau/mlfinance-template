"""Smoke tests for mlfinance package."""


def test_import() -> None:
    """Test that the package can be imported."""
    import mlfinance

    assert mlfinance.__version__ == "0.1.0"


def test_cli_import() -> None:
    from mlfinance.cli import app

    assert app is not None
