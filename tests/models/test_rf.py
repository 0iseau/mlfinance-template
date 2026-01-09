import pandas as pd
import pytest

from mlfinance.models import Random_Forest_Model


def test_rf_target_column_missing() -> None:
    data = pd.DataFrame(
        {
            "Price": [100, 102, 101, 105, 107, 106],
            "High": [101, 103, 101, 105, 108, 106],
            "Low": [99, 100, 100, 104, 106, 105],
            "Volume": [200, 220, 210, 230, 240, 235],
        }
    )

    with pytest.raises(ValueError):
        Random_Forest_Model(data, target="returns")


def test_rf_output() -> None:
    data = pd.DataFrame(
        {
            "Date": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-04",
                "2020-01-05",
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
                "2020-01-09",
                "2020-01-10",
                "2020-01-11",
                "2020-01-12",
                "2020-01-13",
                "2020-01-14",
                "2020-01-15",
                "2020-01-16",
            ],
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

    result, importance, summary = Random_Forest_Model(data, target="returns")

    assert isinstance(result, pd.DataFrame)
    assert "y_true" in result.columns
    assert "y_pred" in result.columns
    assert "difference" in result.columns
    assert "abs_difference" in result.columns
    assert len(result) >= 1

    assert isinstance(importance, pd.DataFrame)
    assert "feature" in importance.columns
    assert "importances_mean" in importance.columns
    assert "importances_std" in importance.columns
    assert len(importance) >= 1

    assert isinstance(summary, dict)
    assert summary["model"] == "Random Forest"
    assert summary["target"] == "returns"
    assert summary["n_train"] >= 1
    assert summary["n_test"] >= 1
    assert summary["n_features"] >= 1
    assert summary["n_estimators"] >= 1


def test_rf_invalid_data_and_target() -> None:
    with pytest.raises(ValueError):
        Random_Forest_Model([1, 2, 3], target="returns")  # type: ignore[arg-type]

    data = pd.DataFrame(
        {
            "Price": [100, 102, 101, 105, 107, 106],
            "High": [101, 103, 101, 105, 108, 106],
            "Low": [99, 100, 100, 104, 106, 105],
            "Volume": [200, 220, 210, 230, 240, 235],
            "Returns_t+1": [0.02, -0.0098, 0.0396, 0.0190, -0.0093, 0.01],
        }
    )

    with pytest.raises(ValueError):
        Random_Forest_Model(data, target="invalid_target")
