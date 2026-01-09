import pandas as pd
import pytest

from mlfinance.models import Ridge_model


def test_ridge_target_column_missing() -> None:
    data = pd.DataFrame(
        {
            "Price": [100, 102, 101, 105, 107, 106],
            "High": [101, 103, 101, 105, 108, 106],
            "Low": [99, 100, 100, 104, 106, 105],
            "Volume": [200, 220, 210, 230, 240, 235],
        }
    )

    with pytest.raises(ValueError):
        Ridge_model(data, alpha=1.0, target="returns")


def test_ridge_output() -> None:
    data = pd.DataFrame(
        {
            "Price": [100, 102, 101, 105, 107, 106],
            "High": [101, 103, 101, 105, 108, 106],
            "Low": [99, 100, 100, 104, 106, 105],
            "Volume": [200, 220, 210, 230, 240, 235],
            "Returns_t+1": [0.02, -0.0098, 0.0396, 0.0190, -0.0093, None],
        }
    )

    result, summary = Ridge_model(data, alpha=1.0, target="returns")

    assert isinstance(result, pd.DataFrame)
    assert "features" in result.columns
    assert "coefficients" in result.columns
    assert "importances_mean" in result.columns
    assert "importances_std" in result.columns

    assert isinstance(summary, dict)
    assert summary["model"] == "Ridge"
    assert summary["target"] == "returns"
    assert summary["alpha"] == 1.0


def test_ridge_invalid_alpha_and_target() -> None:
    data = pd.DataFrame(
        {
            "Price": [100, 102, 101, 105, 107, 106],
            "High": [101, 103, 101, 105, 108, 106],
            "Low": [99, 100, 100, 104, 106, 105],
            "Volume": [200, 220, 210, 230, 240, 235],
            "Returns_t+1": [0.02, -0.0098, 0.0396, 0.0190, -0.0093, None],
        }
    )

    with pytest.raises(ValueError):
        Ridge_model(data, alpha=-1.0, target="returns")

    with pytest.raises(ValueError):
        Ridge_model(data, alpha=1.0, target="invalid_target")
