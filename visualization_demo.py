"""
Example demonstrating PortfolioVisualizer with risk-free rate integration.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from portfolio_analytics.visualization import PortfolioVisualizer
from portfolio_analytics.data_provider import DataProvider

def main():
    """Demonstrate enhanced visualization with risk-free rates."""
    
    print("Portfolio Visualization with Risk-Free Rate Integration Demo")
    print("=" * 60)
    
    # Initialize data provider and visualizer
    print("1. Initializing DataProvider and PortfolioVisualizer...")
    data_provider = DataProvider(cache=True, debug=True)
    visualizer = PortfolioVisualizer(data_provider=data_provider)
    
    # Create sample data
    print("2. Creating sample portfolio data...")
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Generate correlated returns for demonstration
    np.random.seed(42)
    market_returns = np.random.normal(0.0008, 0.015, len(dates))
    
    # Portfolio returns with higher volatility
    portfolio_returns = market_returns * 1.2 + np.random.normal(0, 0.01, len(dates))
    portfolio_returns = pd.Series(portfolio_returns, index=dates, name='Portfolio')
    
    # Benchmark returns
    benchmark_returns = market_returns + np.random.normal(0, 0.008, len(dates))
    benchmark_returns = pd.Series(benchmark_returns, index=dates, name='Benchmark')
    
    # Sample price data
    portfolio_price = (1 + portfolio_returns).cumprod() * 100
    benchmark_price = (1 + benchmark_returns).cumprod() * 100
    price_data = pd.DataFrame({
        'Portfolio': portfolio_price,
        'Benchmark': benchmark_price
    })
    
    # Portfolio weights
    weights = {'Stock A': 0.4, 'Stock B': 0.3, 'Stock C': 0.2, 'Bond': 0.1}
    
    print("3. Demonstrating risk-free rate metadata...")
    if data_provider:
        try:
            risk_free_metadata = data_provider.get_risk_free_rate_metadata()
            print(f"   Risk-free rate symbol: {risk_free_metadata['symbol']}")
            print(f"   Instrument name: {risk_free_metadata['name']}")
            print(f"   Currency: {risk_free_metadata['currency']}")
            print(f"   Frequency: {risk_free_metadata['frequency']}")
        except Exception as e:
            print(f"   Could not retrieve risk-free rate metadata: {e}")
    
    print("4. Available visualization methods with risk-free rate integration:")
    risk_aware_methods = [
        "plot_efficient_frontier (with Capital Allocation Line)",
        "plot_cumulative_returns (with excess returns)",
        "plot_risk_adjusted_metrics (Sharpe, Sortino, Calmar ratios)",
        "create_dashboard (comprehensive risk-adjusted dashboard)"
    ]
    
    for method in risk_aware_methods:
        print(f"   ✓ {method}")
    
    print("\n5. Key enhancements:")
    enhancements = [
        "Automatic risk-free rate fetching from DataProvider",
        "Capital Allocation Line on efficient frontier plots",
        "Excess returns visualization over risk-free rate", 
        "Risk-adjusted Sharpe ratios using actual Treasury rates",
        "Comprehensive risk metrics dashboard",
        "Sortino and Calmar ratio calculations",
        "Risk-free rate overlay on cumulative returns"
    ]
    
    for enhancement in enhancements:
        print(f"   • {enhancement}")
    
    print("\n6. Example usage with risk-free rate:")
    example_code = '''
    # Initialize with DataProvider for risk-free rate access
    data_provider = DataProvider(cache=True)
    visualizer = PortfolioVisualizer(data_provider=data_provider)
    
    # Plot efficient frontier with Capital Allocation Line
    visualizer.plot_efficient_frontier(
        returns_array=returns,
        volatilities_array=volatilities,
        sharpe_ratios_array=sharpe_ratios,
        start_date='2023-01-01',
        end_date='2023-12-31',
        interactive=True
    )
    
    # Plot cumulative returns with excess returns
    visualizer.plot_cumulative_returns(
        returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
        include_excess_returns=True,
        start_date='2023-01-01',
        end_date='2023-12-31',
        interactive=True
    )
    
    # Create comprehensive dashboard with risk-adjusted metrics
    visualizer.create_dashboard(
        portfolio_returns=portfolio_returns,
        price_data=price_data,
        weights=weights,
        benchmark_returns=benchmark_returns,
        start_date='2023-01-01',
        end_date='2023-12-31'
    )
    '''
    
    print(example_code)
    
    print("\nDemo completed! The PortfolioVisualizer now integrates seamlessly")
    print("with the DataProvider to fetch and use risk-free rates for more")
    print("accurate risk-adjusted performance analysis.")

if __name__ == "__main__":
    main()
