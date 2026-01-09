"""Models of machine learning for financial data.

Models:
    - Ridge regression
    - Random Forest Regressor
    - Gradient Boosting
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


def Ridge_model(
    data: pd.DataFrame,
    alpha: float,
    target: str = "returns",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Ridge Regression model training.

    Rdige regression is a linear regression with normalization
    and penalization of large coefficients, often caused by high
    collinearity between features.

    Parameters:
        data (pd.DataFrame): Dataset of prices and indicators indexed by time.
        alpha (float): Also known as lambda, incrementation value of variances in
            variance-covariance matrix X'X (when computing Cov(X,Y)/Var(X)).
        target (str): target for prediction, either returns, direction or rvol.

    Return:
        model (Ridge): trained Ridge model.
        prediction (ndarray): predicted target values.
        summary (dict): summary of model training and performance.

    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if alpha < 0:
        raise ValueError("Lambda coefficient must be greater or equal than 0.")
    if target not in ("returns", "direction", "rvol"):
        raise ValueError("target must be 'returns', 'direction' or 'rvol'.")

    target_cols = {
        "returns": "Returns_t+1",
        "direction": "Direction",
        "rvol": "RVOL_t+1",
    }
    y_col = target_cols[target]
    if y_col not in data.columns:
        raise ValueError(f"Missing target column: {y_col}")

    y = data[y_col]

    # X: drop targets and keep only numeric features (e.g. drop Date strings)
    X = data.drop(columns=list(target_cols.values()), errors="ignore")
    X = X.select_dtypes(include=["number"]).copy()

    # Filter rows where y exists
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    # Drop rows with NaN/inf in X and keep y aligned
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    y = y.loc[X.index]

    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X, y)
    y_pred = model.predict(X)
    perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)

    result = pd.DataFrame(
        {
            "features": ["intercept", *list(X.columns)],
            "coefficients": [model.intercept_, *list(model.coef_)],
            "importances_mean": [np.nan, *list(perm_importance.importances_mean)],
            "importances_std": [np.nan, *list(perm_importance.importances_std)],
        }
    )

    summary = {
        "model": "Ridge",
        "target": target,
        "alpha": alpha,
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "intercept": float(model.intercept_),
        "r2": float(r2_score(y, y_pred)),
        "mae": float(np.mean(np.abs(y - y_pred))),
        "rmse": float(np.sqrt(((y - y_pred) ** 2).mean())),
    }

    return result, summary
