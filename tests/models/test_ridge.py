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
            "Price": [
                100,
                102,
                101,
                105,
                107,
                106,
                108,
                109,
                102,
                110,
                111,
                123,
                120,
                110,
                111,
                115,
            ],
            "High": [
                101,
                103,
                101,
                105,
                108,
                106,
                109,
                110,
                103,
                111,
                112,
                124,
                121,
                111,
                112,
                116,
            ],
            "Low": [
                99,
                100,
                100,
                104,
                106,
                105,
                107,
                108,
                101,
                109,
                110,
                122,
                119,
                109,
                110,
                114,
            ],
            "Volume": [
                200,
                220,
                210,
                230,
                240,
                235,
                245,
                250,
                215,
                255,
                260,
                280,
                275,
                265,
                270,
                290,
            ],
            "Returns_t+1": [
                0.02,
                -0.0098,
                0.0396,
                0.0190,
                -0.0093,
                0.01,
                0.015,
                -0.005,
                0.02,
                0.01,
                -0.01,
                0.03,
                -0.02,
                0.01,
                0.005,
                0.02,
            ],
        }
    )

    result, importance, summary, shap_values = Ridge_model(data, alpha=1.0, target="returns")

    assert isinstance(result, pd.DataFrame)
    assert "y_true" in result.columns
    assert "y_pred" in result.columns

    assert isinstance(importance, pd.DataFrame)
    assert "feature" in importance.columns
    assert "coef" in importance.columns
    assert "importance_mean" in importance.columns
    assert "importance_std" in importance.columns

    assert isinstance(summary, dict)
    assert summary["model"] == "Ridge"
    assert summary["target"] == "returns"
    assert summary["alpha"] == 1.0
    assert shap_values is None or isinstance(shap_values, pd.DataFrame)


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
