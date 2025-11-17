import numpy as np
import pandas as pd
import pytest


# RSI Fixtures
@pytest.fixture
def price_series_inc() -> pd.Series:
    """Fixture that provides a sample increasing price series."""
    return pd.Series(list(range(1, 51)))


@pytest.fixture
def price_series_dec() -> pd.Series:
    """Fixture that provides a sample decreasing price series."""
    return pd.Series(list(range(50, 0, -1)))


@pytest.fixture
def price_series_random() -> pd.Series:
    """Fixture that provides a random price series with realistic financial noise."""
    np.random.seed(42)
    return pd.Series(100 + np.cumsum(np.random.normal(0, 1, 100)))


@pytest.fixture
def price_series_constant() -> pd.Series:
    """Fixture that provides a constant price series."""
    return pd.Series([100] * 50)
