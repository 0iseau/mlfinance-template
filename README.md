# MLFinance – Machine Learning for Financial Prediction

A teaching-size library and CLI for ML-based financial forecasting
and backtesting.

## Project objective

How do machine learning methods effectively apply to financial time series data,
and which practical and methodological challenges arise in this context?

## Setup

### Prerequisites

- Python >= 3.10
- NumPy (≥ 1.26), pandas (≥ 2.1)
- scikit-learn (≥ 1.4), Typer (≥ 0.12)
- `uv`

### `uv` + extras

From the project root:

```bash
uv sync

uv run mlfinance --help
```

Development dependencies (tests, lint, type-check):

```bash
uv sync --extra dev

make check
make test
```

Optional explainability dependencies (avoids SHAP library conflicts):

```bash
uv sync --extra explain

uv sync --extra dev --extra explain
```

Optional TA-Lib dependency (only needed for TA-Lib parity checks):

```bash
uv sync --extra talib

# or combine
uv sync --extra dev --extra talib
```

---

## Usage

```bash
# creates dataset with indicators, summary or filters usable data
mlfinance features data/raw/gold_price.csv \
 --analyze  # or --summary or --build
# Output : results/out_features.csv, csv file with corresponding data

# train model and predict
mlfinance train data/gold_price.csv \
 --target returns \
 --model rf \
 --features all \
 --shap  # or ignore --shap
# --target (returns, direction, rvol)
# --model (ridge, rf, gb)
# --features (all, techincal, stat)
# Output:
# - results/out_results.csv (prediction vs. reality)
# - results/out_importance.csv (permutation importance)
# - printed training summary
# - if --shap: results/out_shap_values.csv
# displays shap values of trained indicators.

# backtest strategy
mlfinance backtest data/gold_price.csv \
 --strategy ml-pred  # or ma-cross
# Output: results/out_backtest.csv

```

---

## 📦 What's included

- **Feature engineering**: Technical indicators, lags, rolling statistics
- **Models**: Random Forest, Gradient Boosting, Linear models
- **Backtesting**: Walk-forward validation, performance metrics
- **CLI and library**: Use from command line or as a Python package
- **Full test coverage**: 80%+ with pytest and hypothesis
- **Type safety**: Strict MyPy configuration
- **Code quality**: Ruff linting and formatting

---

## 📊 Project Structure

```text
src/mlfinance/     # Main package
├── __init__.py    # Public API exports
├── backtest.py    # Backtesting engine
├── cli.py         # CLI interface
├── features.py    # Feature engineering
├── math_utils.py  # Maths helper functions
├── models.py      # ML models
└── validation.py  # Data validation helper function

tests/
└── test_cli_features.py  # Test CLI behavior
└── test_smoke.py         # Smoke tests

features        # Test suite for features
└── test_*.py

maths_utils     # Test suite for maths utils
└── test_*.py

models          # Test suite for models
└── test_*.py

```

## References

## Primary sources

Murphy, J. J. (1999). _Technical Analysis of the Financial Markets_.
New York Institute of Finance.
Wikipedia
B. T. Kelly, D. Xiu (2023). *Financial Machine Learning*

## Technical Documentation

Tradingview Wiki - <https://www.tradingview.com/support/knowledge-base/>
Investopedia Dictionary -
<https://www.investopedia.com/financial-term-dictionary-4769738>
