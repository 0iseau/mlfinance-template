import pandas as pd


def validate_prices(prices: pd.Series[float]) -> pd.Series[float]:
    """Validate the Prices series for financial indicators.

    Parameters
        prices (pd.Series[float]): Series of prices.

    Returns:
        pd.Series[float]: Validated Prices series.

    Raises:
        Error: If Prices not formatted correctly.
    """
    if not isinstance(prices, pd.Series):
        raise TypeError("Prices must be a pandas Series.")

    if not pd.api.types.is_numeric_dtype(prices):
        raise ValueError("Prices series must contain numeric values only.")

    return prices.astype(float)
