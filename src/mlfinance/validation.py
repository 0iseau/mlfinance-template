import pandas as pd


def validate_data(data: pd.Series[float]) -> pd.Series[float]:
    """Validate the Data series for financial indicators.

    Parameters
        data (pd.Series[float]): Series of prices.

    Returns:
        pd.Series[float]: Validated Data series.

    Raises:
        Error: If Data not formatted correctly.
    """
    if not isinstance(data, pd.Series):
        raise TypeError("Data must be a pandas Series.")

    if not pd.api.types.is_numeric_dtype(data):
        raise ValueError("Data series must contain numeric values only.")

    return data.astype(float)
