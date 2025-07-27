# Portfolio Analytics AI 📈

A comprehensive Python package for portfolio analytics, optimization, and risk management using modern portfolio theory and advanced statistical methods.

## 🚀 Features

- **Portfolio Construction & Management**: Create and manage multi-asset portfolios with flexible weighting schemes
- **Modern Portfolio Theory**: Implement mean-variance optimization with efficient frontier calculation
- **Risk Analysis**: Value at Risk (VaR), Expected Shortfall, Maximum Drawdown, and more
- **Performance Analytics**: Comprehensive performance metrics with benchmark comparison
- **Data Integration**: Seamless integration with Yahoo Finance and extensible data provider architecture
- **Visualization**: Interactive and static charts for portfolio analysis
- **Risk Models**: Multiple VaR calculation methods (historical, parametric, Monte Carlo)

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Virtual Environment
```bash
# Clone or navigate to the project directory
cd PortfolioAnalyticsAI

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## 🎯 Quick Start

### Basic Usage

```python
from portfolio_analytics import Portfolio, DataProvider

# Create a data provider
data_provider = DataProvider()

# Define your portfolio
symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
weights = {'AAPL': 0.3, 'GOOGL': 0.2, 'MSFT': 0.2, 'AMZN': 0.15, 'TSLA': 0.15}

# Create portfolio
portfolio = Portfolio(
    name="Tech Portfolio",
    symbols=symbols,
    weights=weights,
    data_provider=data_provider
)

# Load historical data
portfolio.load_data(start_date='2020-01-01', end_date='2023-01-01')

# Get basic metrics
print(f"Annual Return: {portfolio.annual_return():.2%}")
print(f"Annual Volatility: {portfolio.annual_volatility():.2%}")
print(f"Sharpe Ratio: {portfolio.sharpe_ratio():.3f}")
```

### Portfolio Optimization

```python
from portfolio_analytics import PortfolioOptimizer

# Create optimizer
optimizer = PortfolioOptimizer(portfolio)

# Optimize for maximum Sharpe ratio
optimal_weights = optimizer.optimize('max_sharpe')

# Update portfolio with optimal weights
portfolio.update_weights(optimal_weights)

print("Optimized Portfolio Metrics:")
print(f"Annual Return: {portfolio.annual_return():.2%}")
print(f"Annual Volatility: {portfolio.annual_volatility():.2%}")
print(f"Sharpe Ratio: {portfolio.sharpe_ratio():.3f}")
```

### Risk Analysis

```python
from portfolio_analytics import RiskModel

# Create risk model
risk_model = RiskModel(portfolio)

# Calculate Value at Risk
var_95 = risk_model.value_at_risk(confidence_level=0.95)
var_99 = risk_model.value_at_risk(confidence_level=0.99)

# Calculate Expected Shortfall
es_95 = risk_model.expected_shortfall(confidence_level=0.95)

# Calculate Maximum Drawdown
max_dd = risk_model.maximum_drawdown()

print(f"VaR (95%): {var_95:.2%}")
print(f"VaR (99%): {var_99:.2%}")
print(f"Expected Shortfall (95%): {es_95:.2%}")
print(f"Maximum Drawdown: {max_dd:.2%}")
```

## 📊 Visualization

```python
from portfolio_analytics import PortfolioVisualizer

# Create visualizer
visualizer = PortfolioVisualizer(portfolio)

# Plot price history
visualizer.plot_price_history()

# Plot correlation matrix
visualizer.plot_correlation_matrix()

# Plot efficient frontier
visualizer.plot_efficient_frontier()

# Create interactive dashboard
visualizer.create_dashboard()
```

## 🧪 Running Examples

### Console Example
```bash
python example.py
```

### Jupyter Notebook
```bash
jupyter notebook notebooks/01_getting_started.ipynb
```

## 🧪 Testing

Run the test suite to ensure everything is working correctly:

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_portfolio.py -v
```

## 📁 Project Structure

```
PortfolioAnalyticsAI/
├── src/
│   └── portfolio_analytics/
│       ├── __init__.py           # Package initialization
│       ├── portfolio.py          # Core Portfolio class
│       ├── data_provider.py      # Data fetching utilities
│       ├── optimization.py       # Portfolio optimization
│       ├── risk_models.py        # Risk analysis tools
│       ├── performance.py        # Performance analytics
│       └── visualization.py      # Charting and visualization
├── tests/
│   └── test_portfolio.py         # Unit tests
├── notebooks/
│   └── 01_getting_started.ipynb  # Interactive examples
├── example.py                    # Complete usage example
├── requirements.txt              # Package dependencies
├── pyproject.toml               # Project configuration
└── README.md                    # This file
```

## 📈 Key Classes

### Portfolio
Core class for portfolio management and basic analytics.

### DataProvider
Handles data fetching from multiple sources (Yahoo Finance, etc.).

### PortfolioOptimizer
Implements modern portfolio theory optimization techniques.

### RiskModel
Provides comprehensive risk analysis capabilities.

### PerformanceAnalyzer
Calculates performance metrics and benchmark comparisons.

### PortfolioVisualizer
Creates static and interactive visualizations.

## 🔧 Dependencies

- **numpy**: Numerical computations
- **pandas**: Data manipulation and analysis
- **matplotlib**: Static plotting
- **seaborn**: Statistical visualizations
- **plotly**: Interactive charts
- **scikit-learn**: Machine learning utilities
- **yfinance**: Yahoo Finance data
- **scipy**: Scientific computing
- **jupyter**: Interactive notebooks

## 📋 Development

### Code Style
This project uses Black for code formatting:
```bash
black src/ tests/
```

### Type Checking
MyPy is used for static type checking:
```bash
mypy src/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- Create an issue for bug reports or feature requests
- Check the examples and documentation for common use cases
- Review the test suite for implementation details

## 🎯 Future Enhancements

- [ ] Additional data sources (Alpha Vantage, IEX Cloud)
- [ ] Machine learning-based risk models
- [ ] Factor model implementation
- [ ] Real-time portfolio monitoring
- [ ] Advanced optimization algorithms
- [ ] ESG scoring integration
- [ ] Options strategies analysis

---

**Happy Portfolio Analyzing! 📊💼**
