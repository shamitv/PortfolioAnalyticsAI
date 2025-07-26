<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Portfolio Analytics AI - Copilot Instructions

## Project Overview
This is a comprehensive Python package for AI-powered portfolio analytics, optimization, and risk management. The package provides tools for portfolio construction, performance analysis, risk assessment, and visualization.

## Code Style and Standards
- Follow PEP 8 style guidelines for Python code
- Use type hints for all function parameters and return values
- Write comprehensive docstrings in Google style
- Maintain modular code structure with clear separation of concerns
- Use meaningful variable and function names
- Include error handling and input validation

## Project Structure
```
src/portfolio_analytics/
├── __init__.py          # Package initialization and exports
├── portfolio.py         # Core Portfolio class
├── data_provider.py     # Data fetching and management
├── optimization.py      # Portfolio optimization algorithms
├── risk_models.py       # Risk modeling and VaR calculations
├── performance.py       # Performance metrics and analysis
└── visualization.py     # Plotting and visualization tools
```

## Key Dependencies
- **numpy**: Numerical computations and array operations
- **pandas**: Data manipulation and time series handling
- **matplotlib/seaborn**: Static plotting and visualization
- **plotly**: Interactive charts and dashboards
- **scipy**: Optimization algorithms and statistical functions
- **yfinance**: Financial data fetching from Yahoo Finance
- **scikit-learn**: Machine learning models and preprocessing

## Architecture Patterns
- Use composition over inheritance where possible
- Implement abstract base classes for extensibility
- Follow the single responsibility principle
- Use dependency injection for data providers
- Implement proper error handling and logging

## Financial Domain Knowledge
- **Modern Portfolio Theory**: Focus on risk-return optimization
- **Risk Metrics**: VaR, Expected Shortfall, Maximum Drawdown, Sharpe Ratio
- **Performance Analysis**: Alpha, Beta, Tracking Error, Information Ratio
- **Portfolio Optimization**: Mean-variance optimization, risk parity, Black-Litterman
- **Data Handling**: Handle missing data, outliers, and market holidays appropriately

## Testing Guidelines
- Write unit tests for all core functionality
- Mock external data sources in tests
- Test edge cases and error conditions
- Use pytest for test framework
- Maintain test coverage above 80%

## Documentation Standards
- Include comprehensive docstrings for all public methods
- Provide usage examples in docstrings
- Create Jupyter notebooks for tutorials and examples
- Document mathematical formulations and assumptions
- Include references to financial theory and literature

## Performance Considerations
- Vectorize operations using numpy/pandas where possible
- Cache expensive calculations (covariance matrices, etc.)
- Use efficient data structures for time series data
- Consider memory usage for large datasets
- Profile code for optimization bottlenecks

## Security and Data Privacy
- Never hardcode API keys or credentials
- Validate all user inputs
- Handle financial data with appropriate security measures
- Follow best practices for handling sensitive information

## Specific Implementation Notes
- Use pandas DatetimeIndex for all time series data
- Implement proper handling of trading days vs calendar days
- Support multiple data frequencies (daily, weekly, monthly)
- Provide both static and interactive visualization options
- Make optimization algorithms configurable with different methods
- Support multiple risk models and calculation methods
