import pandas as pd


def validate_data(data: pd.Series) -> pd.Series[float]:
    """Validate the Data series for financial indicators.

    Parameters
        data (pd.Series[float]): Series of data.

    Returns:
        pd.Series[float]: Validated Data series.

    Raises:
        Error: If Data not formatted correctly.
    """
    if not isinstance(data, pd.Series):
        raise TypeError("Data must be a pandas Series.")

    converted = pd.to_numeric(data, errors="coerce")

    # Reject non-numeric values (keep NaNs as missing data).
    # If something was not NA but became NA after conversion, it was non-numeric.
    if (data.notna() & converted.isna()).any():
        raise ValueError("Data series must contain numeric values only.")

    return converted.astype(float)
