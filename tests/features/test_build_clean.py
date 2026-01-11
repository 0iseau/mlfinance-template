import pandas as pd
import pytest

from mlfinance.features import _build_clean, rvol


def test_build_clean() -> None:
    data = pd.DataFrame(
        {
            "CLOSE": [100, 102, 101, 105, 107, 106],
            "volume": [200, 220, 210, 230, 240, 235],
            "H": [101, 103, 101, 105, 108, 106],
            "l": [99, 100, 100, 104, 106, 105],
            "other": [1, 2, 3, 4, 5, 6],
        }
    )

    expected = pd.DataFrame(
        {
            "Price": [100, 102, 101, 105, 107, 106],
            "High": [101, 103, 101, 105, 108, 106],
            "Low": [99, 100, 100, 104, 106, 105],
            "Volume": [200, 220, 210, 230, 240, 235],
        }
    )

    expected["Returns_t+1"] = data["CLOSE"].pct_change().shift(-1)
    expected["Direction"] = (expected["Returns_t+1"] > 0).astype(int)
    expected["RVOL_t+1"] = rvol(data["volume"], window=20).shift(-1)

    result = _build_clean(data)

    assert result.equals(expected)

    def test_build_clean_usable_data() -> None:
        data = pd.DataFrame(
            {
                "volume": [200, 220, 210, 230, 240, 235],
                "H": [101, 103, 101, 105, 108, 106],
                "l": [99, 100, 100, 104, 106, 105],
                "other": [1, 2, 3, 4, 5, 6],
            }
        )

        _build_clean(data)

        with pytest.raises(ValueError, match=r"Cannot find usable price column in DataFrame\."):
            _build_clean(data)

    data2 = pd.DataFrame(
        {
            "volume": [200, 220, 210, 230, 240, 235],
            "Open": [100, 102, 101, 105, 107, 106],
            "H": [101, 103, 101, 105, 108, 106],
            "l": [99, 100, 100, 104, 106, 105],
            "other": [1, 2, 3, 4, 5, 6],
        }
    )

    expected2 = pd.DataFrame(
        {
            "Price": [100, 102, 101, 105, 107, 106],
            "High": [101, 103, 101, 105, 108, 106],
            "Low": [99, 100, 100, 104, 106, 105],
            "Volume": [200, 220, 210, 230, 240, 235],
        }
    )

    expected2["Returns_t+1"] = data2["Open"].pct_change().shift(-1)
    expected2["Direction"] = (expected2["Returns_t+1"] > 0).astype(int)
    expected2["RVOL_t+1"] = rvol(data2["volume"], window=20).shift(-1)

    result2 = _build_clean(data2)

    assert result2.equals(expected2)

    data3 = pd.DataFrame(
        {
            "Open": [100, 102, 101, 105, 107, 106],
            "H": [101, 103, 101, 105, 108, 106],
            "l": [99, 100, 100, 104, 106, 105],
            "other": [1, 2, 3, 4, 5, 6],
        }
    )

    expected3 = pd.DataFrame(
        {
            "Price": [100, 102, 101, 105, 107, 106],
            "High": [101, 103, 101, 105, 108, 106],
            "Low": [99, 100, 100, 104, 106, 105],
        }
    )

    expected3["Returns_t+1"] = data3["Open"].pct_change().shift(-1)
    expected3["Direction"] = (expected3["Returns_t+1"] > 0).astype(int)

    result3 = _build_clean(data3)

    assert result3.equals(expected3)
