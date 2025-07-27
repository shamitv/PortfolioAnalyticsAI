# Portfolio Analytics AI 📈

![PyPI](https://img.shields.io/pypi/v/portfolio-analytics-ai)
![Python](https://img.shields.io/pypi/pyversions/portfolio-analytics-ai)
![License](https://img.shields.io/pypi/l/portfolio-analytics-ai)
![Downloads](https://img.shields.io/pypi/dm/portfolio-analytics-ai)

## ⚠️ Project Status

**This project is currently a Proof of Concept (POC) and is in very early stages of development.**

**Important Disclaimers:**
- 🚧 **Early Development**: This is a POC implementation and should be considered experimental
- 📊 **Unvalidated**: The algorithms and calculations have not been validated with real-world financial data
- 👨‍💼 **No Professional Review**: The financial models and risk calculations have not been reviewed by finance professionals
- 🚫 **Not Production Ready**: This package is **NOT ready for production use** or real investment decisions

**Use at your own risk. This software is intended for educational and research purposes only.**

---

A comprehensive Python package for AI-powered portfolio analytics, optimization, and risk management. Built on modern portfolio theory and advanced statistical methods, this package provides professional-grade tools for investment analysis, portfolio construction, and risk assessment.

## 🌍 Where It Can Be Used

**Portfolio Analytics AI** is designed for a wide range of financial applications:

### Investment Management
- **Asset Managers**: Portfolio construction, optimization, and performance monitoring
- **Wealth Management**: Client portfolio analysis and risk assessment
- **Hedge Funds**: Quantitative trading strategies and risk management
- **Pension Funds**: Long-term portfolio optimization and liability matching

### Financial Research & Academia
- **Quantitative Research**: Backtesting investment strategies and factor analysis
- **Academic Research**: Financial modeling and portfolio theory validation
- **Risk Management**: Institution-wide risk assessment and stress testing
- **Financial Education**: Teaching modern portfolio theory and investment concepts

### Fintech & Investment Platforms
- **Robo-Advisors**: Automated portfolio allocation and rebalancing
- **Investment Apps**: Portfolio analytics and performance tracking
- **Trading Platforms**: Risk assessment and optimization tools
- **Financial Advisory**: Client portfolio recommendations and reporting

### Personal Finance
- **Individual Investors**: Portfolio optimization and risk analysis
- **Financial Advisors**: Client portfolio construction and monitoring
- **Retirement Planning**: Long-term investment strategy optimization

## 🚀 Installation

### From PyPI (Recommended)
```bash
pip install portfolio-analytics-ai
```

### From Source
```bash
git clone https://github.com/shamitv/PortfolioAnalyticsAI.git
cd PortfolioAnalyticsAI
pip install -e .
```

## 📁 Project Structure

```
PortfolioAnalyticsAI/
├── src/portfolio_analytics/           # Main package directory
│   ├── __init__.py                   # Package initialization and exports
│   ├── portfolio.py                  # Core Portfolio class
│   ├── data_provider.py              # Data fetching and management
│   ├── optimization.py               # Portfolio optimization algorithms
│   ├── risk_models.py                # Risk modeling and VaR calculations
│   ├── performance.py                # Performance metrics and analysis
│   ├── visualization.py              # Plotting and visualization tools
│   ├── analyzer.py                   # Comprehensive portfolio analyzer
│   └── sample_data/                  # Sample datasets and market data
├── notebooks/                        # Jupyter notebook examples
│   ├── 01_getting_started.ipynb      # Basic usage and introduction
│   ├── 02_analyzer_demo.ipynb        # Portfolio analyzer demonstration
│   ├── 03_cache_population_demo.ipynb # Data caching examples
│   └── 04_random_portfolio_analysis.ipynb # Advanced analysis examples
├── tests/                            # Unit tests and integration tests
├── sample_data/                      # External sample data files
├── requirements.txt                  # Package dependencies
├── pyproject.toml                    # Package configuration
└── README.md                         # Project documentation
```

## 🏗️ Core Classes Overview

### 1. **Portfolio** (`portfolio.py`)
The central class for portfolio management and analysis.

**Key Features:**
- Portfolio construction with custom weights or equal weighting
- Performance metrics calculation (returns, volatility, Sharpe ratio)
- Integration with data providers for historical data loading
- Portfolio rebalancing and weight optimization
- Risk-adjusted performance analysis

**Primary Methods:**
- `load_data()`: Load historical price data
- `portfolio_return()`, `annual_return()`: Calculate returns
- `portfolio_volatility()`, `annual_volatility()`: Calculate risk metrics
- `sharpe_ratio()`: Risk-adjusted performance
- `optimize_weights()`: Portfolio optimization

### 2. **DataProvider** (`data_provider.py`)
Comprehensive data fetching and management system.

**Key Features:**
- Yahoo Finance integration for real-time and historical data
- SQLite caching for improved performance
- Risk-free rate data (Treasury yields)
- Market calendars and trading day handling
- Support for stocks, ETFs, and sector data

**Primary Methods:**
- `get_price_data()`: Fetch historical price data
- `get_risk_free_rate()`: Retrieve Treasury yield data
- `cache_stock_data()`: Cache data for offline use
- `get_sector_etfs()`: Access sector ETF information
- `get_sp500_companies()`: S&P 500 constituent data

### 3. **PortfolioOptimizer** (`optimization.py`)
Advanced portfolio optimization using modern portfolio theory.

**Key Features:**
- Mean-variance optimization (Markowitz)
- Maximum Sharpe ratio optimization
- Minimum variance portfolios
- Target return optimization
- Efficient frontier calculation
- Custom constraints and bounds support

**Primary Methods:**
- `optimize()`: Main optimization function
- `calculate_efficient_frontier()`: Generate efficient frontier
- `max_sharpe_optimization()`: Find maximum Sharpe ratio portfolio
- `min_variance_optimization()`: Find minimum variance portfolio

### 4. **RiskModel** (`risk_models.py`)
Comprehensive risk modeling and Value-at-Risk calculations.

**Key Features:**
- Multiple VaR calculation methods (Historical, Parametric, Monte Carlo)
- Expected Shortfall (Conditional VaR)
- Maximum Drawdown analysis
- Downside risk metrics
- Portfolio-level and component-level risk attribution

**Primary Methods:**
- `calculate_var()`: Value at Risk calculation
- `calculate_expected_shortfall()`: Expected Shortfall
- `calculate_maximum_drawdown()`: Maximum drawdown analysis
- `calculate_downside_deviation()`: Downside risk metrics
- `calculate_component_var()`: Risk decomposition

### 5. **PerformanceAnalyzer** (`performance.py`)
Portfolio performance analysis and benchmarking.

**Key Features:**
- Comprehensive performance metrics
- Benchmark comparison and relative performance
- Risk-adjusted return measures
- Rolling performance analysis
- Attribution analysis

**Primary Methods:**
- `calculate_metrics()`: Comprehensive performance metrics
- `alpha_beta_analysis()`: Market risk analysis
- `tracking_error()`: Benchmark tracking metrics
- `information_ratio()`: Risk-adjusted excess returns
- `rolling_metrics()`: Time-varying performance analysis

### 6. **PortfolioVisualizer** (`visualization.py`)
Professional-grade visualization tools for portfolio analysis.

**Key Features:**
- Static and interactive charts
- Efficient frontier visualization
- Performance dashboards
- Risk analysis plots
- Customizable styling and themes

**Primary Methods:**
- `plot_efficient_frontier()`: Interactive efficient frontier
- `plot_cumulative_returns()`: Performance over time
- `plot_correlation_matrix()`: Asset correlation heatmap
- `plot_drawdown()`: Drawdown analysis
- `create_dashboard()`: Comprehensive dashboard

### 7. **Analyzer** (`analyzer.py`)
AI-powered comprehensive portfolio analysis system.

**Key Features:**
- Automated portfolio analysis and insights
- LLM-ready visualizations and metrics
- Greeks calculation for options portfolios
- Multi-dimensional risk assessment
- Integrated reporting system

**Primary Methods:**
- `generate_comprehensive_analysis()`: Complete portfolio analysis
- `generate_metrics()`: All performance and risk metrics
- `generate_visualizations()`: Chart generation for analysis
- `generate_insights()`: AI-powered insights and recommendations

## 📚 Sample Notebooks

The `notebooks/` directory contains comprehensive examples and tutorials:

### **01_getting_started.ipynb**
- **Purpose**: Introduction to basic portfolio construction and analysis
- **Content**: Creating portfolios, loading data, basic metrics calculation
- **Audience**: Beginners to portfolio analytics
- **Key Concepts**: Portfolio class usage, data loading, simple optimization

### **02_analyzer_demo.ipynb**
- **Purpose**: Demonstrates the comprehensive Analyzer class
- **Content**: Advanced analytics, visualization generation, AI-powered insights
- **Audience**: Intermediate users seeking comprehensive analysis
- **Key Concepts**: Multi-dimensional analysis, automated reporting, visualization

### **03_cache_population_demo.ipynb**
- **Purpose**: Data caching and performance optimization
- **Content**: Setting up data caches, offline analysis, performance improvements
- **Audience**: Users working with large datasets or limited internet
- **Key Concepts**: Data caching, SQLite integration, performance optimization

### **04_random_portfolio_analysis.ipynb**
- **Purpose**: Advanced portfolio analysis with random portfolios and optimization
- **Content**: Monte Carlo portfolio generation, efficient frontier analysis, risk decomposition
- **Audience**: Advanced users and researchers
- **Key Concepts**: Portfolio simulation, optimization comparison, risk analysis

## 📊 Risk Metrics Calculation

### **Value at Risk (VaR)**
Measures the potential loss in portfolio value over a specific time period at a given confidence level.

**Calculation Methods:**
1. **Historical VaR**: Uses historical return distribution
   ```
   VaR = Percentile of historical returns at (1 - confidence_level)
   ```

2. **Parametric VaR**: Assumes normal distribution of returns
   ```
   VaR = μ - (σ × Z_α)
   where μ = mean return, σ = standard deviation, Z_α = critical value
   ```

3. **Monte Carlo VaR**: Uses simulated return paths
   ```
   VaR = Percentile of simulated returns at (1 - confidence_level)
   ```

### **Expected Shortfall (ES)**
Average loss exceeding the VaR threshold, providing tail risk measurement.
```
ES = E[Loss | Loss > VaR]
```

### **Maximum Drawdown**
Maximum peak-to-trough decline in portfolio value.
```
Drawdown_t = (Peak_value - Current_value) / Peak_value
Max_Drawdown = max(Drawdown_t) for all t
```

### **Sharpe Ratio**
Risk-adjusted return measure.
```
Sharpe_Ratio = (Portfolio_Return - Risk_Free_Rate) / Portfolio_Volatility
```

### **Sortino Ratio**
Downside risk-adjusted return measure.
```
Sortino_Ratio = (Portfolio_Return - Risk_Free_Rate) / Downside_Deviation
```

### **Beta**
Systematic risk relative to market benchmark.
```
Beta = Covariance(Portfolio_Returns, Market_Returns) / Variance(Market_Returns)
```

## 🎨 Visualization Gallery

### **1. Efficient Frontier Plot**
![Efficient Frontier Placeholder](https://via.placeholder.com/600x400/1f77b4/ffffff?text=Efficient+Frontier+Plot)

**Description**: Interactive plot showing the risk-return trade-off for optimal portfolios.

**Usage & Interpretation**:
- **X-axis**: Portfolio volatility (risk)
- **Y-axis**: Expected return
- **Curve**: Represents optimal portfolios at each risk level
- **Points**: Individual assets and current portfolio position
- **Interpretation**: Portfolios on the frontier are optimal; points below are sub-optimal

### **2. Cumulative Returns Chart**
![Cumulative Returns Placeholder](https://via.placeholder.com/600x400/ff7f0e/ffffff?text=Cumulative+Returns+Chart)

**Description**: Time series plot comparing portfolio performance against benchmarks.

**Usage & Interpretation**:
- **X-axis**: Time period
- **Y-axis**: Cumulative return (starting from 1.0 or 100%)
- **Lines**: Portfolio vs benchmark performance
- **Interpretation**: Upward slope indicates positive returns; steeper slope shows better performance

### **3. Correlation Matrix Heatmap**
![Correlation Matrix Placeholder](https://via.placeholder.com/600x400/2ca02c/ffffff?text=Correlation+Matrix+Heatmap)

**Description**: Color-coded matrix showing correlations between portfolio assets.

**Usage & Interpretation**:
- **Color Scale**: Red (negative correlation) to Blue (positive correlation)
- **Values**: Range from -1 (perfect negative) to +1 (perfect positive)
- **Interpretation**: Lower correlations indicate better diversification benefits

### **4. Portfolio Composition Pie Chart**
![Portfolio Composition Placeholder](https://via.placeholder.com/600x400/d62728/ffffff?text=Portfolio+Composition+Chart)

**Description**: Visual representation of portfolio weights and asset allocation.

**Usage & Interpretation**:
- **Sectors**: Different colors represent different assets
- **Size**: Proportional to portfolio weight
- **Interpretation**: Shows concentration risk and diversification level

### **5. Drawdown Analysis**
![Drawdown Analysis Placeholder](https://via.placeholder.com/600x400/9467bd/ffffff?text=Drawdown+Analysis+Chart)

**Description**: Time series showing portfolio drawdowns from peak values.

**Usage & Interpretation**:
- **X-axis**: Time period
- **Y-axis**: Drawdown percentage (negative values)
- **Shaded areas**: Periods of loss from peak
- **Interpretation**: Deeper/longer drawdowns indicate higher risk periods

### **6. Risk-Adjusted Metrics Dashboard**
![Risk Metrics Dashboard Placeholder](https://via.placeholder.com/600x400/8c564b/ffffff?text=Risk+Metrics+Dashboard)

**Description**: Comprehensive dashboard showing multiple risk and performance metrics.

**Usage & Interpretation**:
- **Multiple Panels**: Various risk metrics in organized layout
- **Gauges/Bars**: Visual representation of metric values
- **Benchmarks**: Comparison against market standards
- **Interpretation**: Provides holistic view of portfolio risk profile

### **7. Returns Distribution Histogram**
![Returns Distribution Placeholder](https://via.placeholder.com/600x400/e377c2/ffffff?text=Returns+Distribution+Histogram)

**Description**: Histogram showing the distribution of portfolio returns with normal distribution overlay.

**Usage & Interpretation**:
- **Bars**: Frequency of returns in each range
- **Curve**: Normal distribution overlay
- **Tail Areas**: Extreme loss/gain probabilities
- **Interpretation**: Shows return distribution characteristics and tail risks

### **8. Performance Attribution Chart**
![Performance Attribution Placeholder](https://via.placeholder.com/600x400/7f7f7f/ffffff?text=Performance+Attribution+Chart)

**Description**: Breakdown of portfolio performance by asset or sector contributions.

**Usage & Interpretation**:
- **Bars**: Contribution of each holding to total return
- **Colors**: Positive (green) vs negative (red) contributions
- **Interpretation**: Identifies which holdings drove performance

## 🔧 Quick Start Example

```python
import pandas as pd
from portfolio_analytics import Portfolio, DataProvider, PortfolioOptimizer, PortfolioVisualizer

# Initialize components
data_provider = DataProvider()
visualizer = PortfolioVisualizer()

# Create a technology portfolio
tech_stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
portfolio = Portfolio(tech_stocks, name="Tech Portfolio")

# Load historical data
portfolio.load_data(data_provider, start_date="2020-01-01", end_date="2023-12-31")

# Calculate basic metrics
annual_return = portfolio.annual_return()
annual_vol = portfolio.annual_volatility()
sharpe = portfolio.sharpe_ratio()

print(f"Annual Return: {annual_return:.2%}")
print(f"Annual Volatility: {annual_vol:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")

# Optimize portfolio
optimizer = PortfolioOptimizer()
optimal_weights = optimizer.optimize(portfolio.returns, method="max_sharpe")

# Create visualizations
fig = visualizer.plot_efficient_frontier(portfolio.returns)
fig.show()

# Generate comprehensive analysis
from portfolio_analytics import Analyzer

analyzer = Analyzer(portfolio)
analysis = analyzer.generate_comprehensive_analysis()
```

## 📈 Advanced Features

### Risk Management
- **Multi-method VaR calculation**: Historical, Parametric, Monte Carlo
- **Stress testing**: Scenario analysis and sensitivity testing
- **Risk decomposition**: Component and marginal risk contributions
- **Tail risk analysis**: Expected Shortfall and extreme value analysis

### Portfolio Optimization
- **Multiple objectives**: Return maximization, risk minimization, Sharpe optimization
- **Constraints support**: Weight bounds, sector limits, turnover constraints
- **Robust optimization**: Handling parameter uncertainty
- **Multi-period optimization**: Dynamic rebalancing strategies

### Performance Analysis
- **Attribution analysis**: Performance decomposition by factors
- **Style analysis**: Return-based style analysis
- **Benchmark comparison**: Relative performance metrics
- **Risk-adjusted returns**: Multiple Sharpe-like ratios

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/shamitv/PortfolioAnalyticsAI/issues)
- **Documentation**: [Full documentation](https://github.com/shamitv/PortfolioAnalyticsAI/wiki)
- **PyPI Package**: [portfolio-analytics-ai](https://pypi.org/project/portfolio-analytics-ai/)

## 🎯 Roadmap

### Upcoming Features
- **Machine Learning Integration**: ML-based return predictions and risk modeling
- **Alternative Data Sources**: Integration with additional financial data providers
- **Options Analytics**: Options pricing and Greeks calculation
- **ESG Integration**: Environmental, Social, and Governance factor analysis
- **Real-time Analytics**: Live portfolio monitoring and alerts
- **API Development**: RESTful API for portfolio analytics services

---

**Portfolio Analytics AI** - Empowering investment decisions through advanced analytics and AI-driven insights.

[![GitHub](https://img.shields.io/badge/GitHub-shamitv/PortfolioAnalyticsAI-blue?logo=github)](https://github.com/shamitv/PortfolioAnalyticsAI)
[![PyPI](https://img.shields.io/badge/PyPI-portfolio--analytics--ai-blue?logo=pypi)](https://pypi.org/project/portfolio-analytics-ai/)
