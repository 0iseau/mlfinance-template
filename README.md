# MLFinance – Machine Learning for Financial Prediction

_"A teaching-size library and CLI for ML-based financial forecasting and backtesting."_

This repository is a **template** for the Winter 2025 MSc final project.
Fork or use as a GitHub Template → complete the TODOs → tag **`v1.0.0`**.

---

## ✨ Quick demo

```bash
# editable install
pip install -e .[dev]

# train model and predict
mlfinance train data/features.csv --target returns --model rf
# > Trained RandomForest model
# > Cross-validation R²: 0.42
# > Feature importance saved

# backtest strategy
mlfinance backtest data/prices.csv --strategy ml-momentum
# > Sharpe ratio: 1.35
# > Max drawdown: -12.4%
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

## 🛠️ Development Workflow

```bash
make install-dev    # Set up development environment
make check          # Run all quality checks
make test           # Run tests with coverage
```

---

## 📊 Project Structure

```
src/mlfinance/     # Main package
├── __init__.py    # Public API exports
├── cli.py         # CLI interface
├── features.py    # Feature engineering
├── models.py      # ML models
└── backtest.py    # Backtesting engine

tests/             # Test suite
└── test_*.py      # Test modules
```

---

## 📖 Documentation

See [PROJECT_SPECIFICATION.md](PROJECT_SPECIFICATION.md) for full project requirements.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
