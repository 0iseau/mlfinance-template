"""Models of machine learning for financial data.

Models:
    - Ridge regression
    - Random Forest Regressor
    - Gradient Boosting

Models processes are explained in their respective demonstration docs in models_demo/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


def _data_preparation(
    data: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Helper for data preparation for models."""
    target_cols = {
        "returns": "Returns_t+1",
        "direction": "Direction",
        "rvol": "RVOL_t+1",
    }
    y_col = target_cols[target]
    if y_col not in data.columns:
        raise ValueError(f"Missing target column: {y_col}")

    y = data[y_col]

    # X: drop targets and keep only numeric features
    X = data.drop(columns=list(target_cols.values()), errors="ignore")
    X = X.select_dtypes(include=["number"]).copy()

    # Filter rows where y exists
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    # Drop rows with NaN/inf in X and keep y aligned
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    y = y.loc[X.index]

    # split train/test (80/20)
    n = len(X)
    n_test = max(1, int(0.2 * n))
    X_train, X_test = X.iloc[:-n_test], X.iloc[-n_test:]
    y_train, y_test = y.iloc[:-n_test], y.iloc[-n_test:]

    return X_train, X_test, y_train, y_test


def Ridge_model(
    data: pd.DataFrame,
    alpha: float,
    target: str = "returns",
    features: str = "all",
    shap: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], pd.DataFrame | None]:
    """Ridge Regression model training.

    Rdige regression is a linear regression with normalization
    and penalization of large coefficients, often caused by high
    collinearity between features.

    Parameters:
        data (pd.DataFrame): dataset of prices and indicators indexed by time.
        alpha (float): Also known as lambda, incrementation value of variances in
            variance-covariance matrix X'X (when computing Cov(X,Y)/Var(X)).
        target (str): target for prediction. Either returns, direction or rvol.
        features (str): features used for training. Either technical, stat or all.

    Return:
        result (pd.DataFrame): regression results.
        importance (pd.DataFrame): feature importance.
        summary (dict): summary of model training and performance.
        shap_values (pd.DataFrame | None): SHAP values for feature importance.

    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if alpha < 0:
        raise ValueError("Lambda coefficient must be greater or equal than 0.")
    if target not in ("returns", "direction", "rvol"):
        raise ValueError("target must be 'returns', 'direction' or 'rvol'.")

    X_train, X_test, y_train, y_test = _data_preparation(data, target)

    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if target == "direction":
        y_pred = (y_pred >= 0.5).astype(int)

    shap_values = shap_analysis(X_test, X_train, model) if shap else None

    perm_importance = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=0, n_jobs=-1
    )

    # Predictions
    result = pd.DataFrame(index=X_test.index)

    result["y_true"] = y_test
    result["y_pred"] = y_pred
    result["difference"] = y_test - y_pred
    result["abs_difference"] = np.abs(y_test - y_pred)

    importance = pd.DataFrame(
        {
            "feature": ["intercept", *list(X_train.columns)],
            "coef": [float(model.intercept_), *list(model.coef_)],
            "importance_mean": [np.nan, *list(perm_importance.importances_mean)],
            "importance_std": [np.nan, *list(perm_importance.importances_std)],
        }
    ).sort_values("importance_mean", ascending=False, na_position="last", ignore_index=True)

    summary = {
        "model": "Ridge",
        "target": target,
        "indicators": features,
        "alpha": float(alpha),
        "n_samples": int(len(y_train) + len(y_test)),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_features": int(X_train.shape[1]),
        "intercept": float(model.intercept_),
        "r2": float(r2_score(y_test, y_pred)),
        "mae": float(np.mean(np.abs(y_test - y_pred))),
        "rmse": float(np.sqrt(((y_test - y_pred) ** 2).mean())),
    }

    return result, importance, summary, shap_values


def Random_Forest_Model(
    data: pd.DataFrame,
    target: str = "returns",
    features: str = "all",
    shap: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], pd.DataFrame | None]:
    """Random Forest regression.

    Random forest regression forecasts y(t+1) at yt by "training" on past data.
    Its methodology consists of creating a lot of randomly generated sets of rows from
    our original dataset and train a tree on each one. The final forecast value is the
    mean of all tree values.

    Parameters:
        data (pd.DataFrame): dataset of prices and indicators indexed by time.
        target (str): target for the prediction. Either returns, direction or rvol.
        features (str): features used for training. Either technical, stat or all.

    Returns:
        results (pd.DataFrame): regression results.
        importance (pd.DataFrame): feature importance.
        summary (dict): summary of model training and performance.
        shap_values (pd.DataFrame | None): SHAP values for feature importance.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas dataframe.")
    if target not in {"returns", "direction", "rvol"}:
        raise ValueError("Target must be 'returns', 'direction' or 'rvol'")

    X_train, X_test, y_train, y_test = _data_preparation(data, target)

    # Fit on train, predict on test
    model = RandomForestRegressor(n_jobs=-1, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if target == "direction":
        y_pred = (y_pred >= 0.5).astype(int)

    shap_values = shap_analysis(X_test, X_train, model) if shap else None

    perm_importance = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=0, n_jobs=-1
    )

    result = pd.DataFrame(index=X_test.index)

    result["y_true"] = y_test
    result["y_pred"] = y_pred
    result["difference"] = y_test - y_pred
    result["abs_difference"] = np.abs(y_test - y_pred)

    importance = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importances_mean": perm_importance.importances_mean,
            "importances_std": perm_importance.importances_std,
        }
    ).sort_values("importances_mean", ascending=False)

    summary = {
        "model": "Random Forest",
        "target": target,
        "indicators": features,
        "n_test": len(y_test),
        "n_train": len(y_train),
        "n_features": int(X_test.shape[1]),
        "r2": float(r2_score(y_test, y_pred)),
        "mae": float(np.mean(np.abs(y_test - y_pred))),
        "rmse": float(np.sqrt(((y_test - y_pred) ** 2).mean())),
        "n_estimators": model.n_estimators,
    }

    return result, importance, summary, shap_values


def gradient_boosting_model(
    data: pd.DataFrame,
    target: str = "returns",
    features: str = "all",
    shap: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], pd.DataFrame | None]:
    """Gradient boosting model implementation : XGBoost.

    Gradient boosting predicts y(t+1) at time t by training sequentially on past
    data. Unlike random forest, it trains on the actual dataset, and sequentially
    builds trees that try to adjust the residuals of previous trees. A prediction
    is the result of all adjustments made by each trees after filtering the row through
    each rule.

    Parameters:
        data (pd.DataFrame): dataset of prices and indicators indexed by time.
        target (str): target for the prediction. Either returns, direction or rvol.

    Returns:
        results (pd.DataFrame): regression results.
        importance (pd.DataFrame): feature importance.
        summary (dict): summary of model training and performance.
        shap_values (pd.DataFrame | None): SHAP values for feature importance.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas data frame.")
    if target not in {"returns", "direction", "rvol"}:
        raise ValueError("Target must be 'returns', 'direction' or 'rvol'.")

    X_train, X_test, y_train, y_test = _data_preparation(data, target)

    model = GradientBoostingRegressor(learning_rate=0.05, n_estimators=300, random_state=0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if target == "direction":
        y_pred = (y_pred >= 0.5).astype(int)

    shap_values = shap_analysis(X_test, X_train, model) if shap else None

    perm_importance = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=0, n_jobs=-1
    )

    result = pd.DataFrame(index=X_test.index)

    result["y_true"] = y_test
    result["y_pred"] = y_pred
    result["difference"] = y_test - y_pred
    result["abs_difference"] = np.abs(y_test - y_pred)

    importance = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importances_mean": perm_importance.importances_mean,
            "importances_std": perm_importance.importances_std,
        }
    ).sort_values("importances_mean", ascending=False)

    summary = {
        "model": "Gradient Boosting",
        "target": target,
        "indicators": features,
        "n_test": len(y_test),
        "n_train": len(y_train),
        "n_features": int(X_test.shape[1]),
        "r2": float(r2_score(y_test, y_pred)),
        "mae": float(np.mean(np.abs(y_test - y_pred))),
        "rmse": float(np.sqrt(((y_test - y_pred) ** 2).mean())),
        "n_estimators": model.n_estimators,
        "learning_rate": model.learning_rate,
        "max_depth": model.max_depth,
    }

    return result, importance, summary, shap_values


def shap_analysis(x_test: pd.DataFrame, x_train: pd.DataFrame, model: object) -> pd.DataFrame:
    """Placeholder for SHAP analyses function."""
    try:
        import shap  # type: ignore[import-not-found]
    except Exception as e:
        raise ImportError("Install shap with >uv sync --extra explain") from e

    name = model.__class__.__name__.lower()

    if "ridge" in name:
        explainer = shap.LinearExplainer(model, x_train)
        shap_values = explainer.shap_values(x_test)
    else:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_test)

    return pd.DataFrame(shap_values, index=x_test.index, columns=x_test.columns)
