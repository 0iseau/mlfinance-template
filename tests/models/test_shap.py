from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from mlfinance.models import shap_analysis


def test_shap_analysis_ridge_returns_dataframe_with_expected_shape() -> None:
    pytest.importorskip("shap")

    rng = np.random.default_rng(0)

    x_train = pd.DataFrame(
        rng.normal(size=(40, 3)),
        columns=["f1", "f2", "f3"],
    )
    y_train = pd.Series(rng.normal(size=40), name="y")

    model = Ridge(alpha=1.0)
    model.fit(x_train, y_train)

    x_test = pd.DataFrame(
        rng.normal(size=(5, 3)),
        columns=x_train.columns,
    )

    shap_df = shap_analysis(x_test=x_test, x_train=x_train, model=model)

    assert isinstance(shap_df, pd.DataFrame)
    assert shap_df.shape == x_test.shape
    assert list(shap_df.columns) == list(x_test.columns)
    assert list(shap_df.index) == list(x_test.index)
