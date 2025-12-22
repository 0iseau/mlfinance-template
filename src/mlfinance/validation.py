import pandas as pd


def validate_prices(Prices: pd.Series[float]) -> pd.Series[float]:
    """Validate the Prices series for financial indicators.

    Parameters
        Prices (pd.Series[float]): Series of prices.

    Returns:
        pd.Series[float]: Validated Prices series.

    Raises:
        Error: If Prices not formatted correctly.
    """
    if not isinstance(Prices, pd.Series):
        raise TypeError("Prices must be a pandas Series.")

    if not pd.api.types.is_numeric_dtype(Prices):
        raise ValueError("Prices series must contain numeric values only.")

    if Prices.empty:
        # Able to handle empty series by returning it directly
        return Prices.astype(float)

    return Prices
