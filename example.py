#!/usr/bin/env python3
"""
Simple example script to test the Portfolio Analytics AI package.

This script demonstrates basic functionality without requiring internet access.
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add the src directory to Python path for imports
sys.path.insert(0, 'src')

from portfolio_analytics import Portfolio, DataProvider
from portfolio_analytics.optimization import PortfolioOptimizer
from portfolio_analytics.risk_models import RiskModel
from portfolio_analytics.performance import PerformanceAnalyzer


def create_sample_data(symbols, start_date='2020-01-01', periods=1000):
    """Create sample price data for testing."""
    print("Creating sample data...")
    
    # Create date range
    dates = pd.date_range(start=start_date, periods=periods, freq='D')
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Generate random returns with different characteristics for each asset
    returns_data = {}
    base_return = 0.0001  # Small daily return
    
    for i, symbol in enumerate(symbols):
        # Each asset has slightly different volatility and drift
        volatility = 0.015 + i * 0.005  # Increasing volatility
        drift = base_return + i * 0.00005  # Slightly different expected returns
        
        # Generate random returns
        returns = np.random.normal(drift, volatility, periods)
        
        # Create price series starting at 100
        prices = [100]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        returns_data[symbol] = prices
    
    # Create DataFrame
    price_data = pd.DataFrame(returns_data, index=dates)
    
    print(f"Sample data created: {price_data.shape}")
    print(f"Date range: {price_data.index[0]} to {price_data.index[-1]}")
    
    return price_data


def main():
    """Main function to demonstrate portfolio analytics functionality."""
    print("=" * 60)
    print("Portfolio Analytics AI - Example Usage")
    print("=" * 60)
    
    # Define portfolio symbols
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    print(f"Portfolio symbols: {symbols}")
    
    # Create sample data (simulating historical prices)
    sample_data = create_sample_data(symbols)
    
    # Create portfolio with equal weights
    print("\n1. Creating Portfolio...")
    portfolio = Portfolio(symbols, name="Tech Portfolio")
    print(f"Created: {portfolio}")
    print(f"Initial weights: {dict(zip(symbols, portfolio.weights))}")
    
    # Manually set the data (normally would come from DataProvider)
    portfolio.data = sample_data
    portfolio.returns = sample_data.pct_change().dropna()
    
    print(f"Loaded data: {portfolio.data.shape[0]} days")
    
    # Calculate basic metrics
    print("\n2. Basic Portfolio Metrics...")
    try:
        annual_return = portfolio.annual_return()
        annual_volatility = portfolio.annual_volatility()
        sharpe_ratio = portfolio.sharpe_ratio()
        var_95 = portfolio.calculate_var(0.95)
        
        print(f"Annual Return: {annual_return:.2%}")
        print(f"Annual Volatility: {annual_volatility:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
        print(f"Value at Risk (95%): {var_95:.2%}")
    except Exception as e:
        print(f"Error calculating metrics: {e}")
    
    # Get correlation matrix
    print("\n3. Asset Correlations...")
    try:
        correlation_matrix = portfolio.get_correlation_matrix()
        print("Correlation Matrix:")
        print(correlation_matrix.round(3))
    except Exception as e:
        print(f"Error calculating correlations: {e}")
    
    # Portfolio optimization
    print("\n4. Portfolio Optimization...")
    try:
        original_weights = portfolio.weights.copy()
        
        # Try to optimize (this might fail if scipy is not available)
        try:
            optimized_weights = portfolio.optimize(method="max_sharpe")
            
            print("Optimization Results:")
            for i, symbol in enumerate(symbols):
                print(f"{symbol}: {original_weights[i]:.3f} -> {optimized_weights[i]:.3f}")
            
            # Calculate new metrics
            new_return = portfolio.annual_return()
            new_volatility = portfolio.annual_volatility()
            new_sharpe = portfolio.sharpe_ratio()
            
            print(f"\nOptimized Portfolio Metrics:")
            print(f"Annual Return: {new_return:.2%}")
            print(f"Annual Volatility: {new_volatility:.2%}")
            print(f"Sharpe Ratio: {new_sharpe:.3f}")
            
        except ImportError:
            print("Optimization requires scipy package. Skipping optimization.")
        except Exception as e:
            print(f"Optimization failed: {e}")
            
    except Exception as e:
        print(f"Error in optimization section: {e}")
    
    # Risk analysis
    print("\n5. Risk Analysis...")
    try:
        portfolio_returns = (portfolio.returns * portfolio.weights).sum(axis=1)
        risk_model = RiskModel()
        
        # Calculate various risk metrics
        var_99 = risk_model.calculate_var(portfolio_returns, 0.99)
        expected_shortfall = risk_model.calculate_expected_shortfall(portfolio_returns, 0.95)
        max_drawdown_info = risk_model.calculate_maximum_drawdown(portfolio_returns)
        
        print(f"VaR (99%): {var_99:.2%}")
        print(f"Expected Shortfall (95%): {expected_shortfall:.2%}")
        print(f"Maximum Drawdown: {max_drawdown_info['max_drawdown']:.2%}")
        
    except Exception as e:
        print(f"Error in risk analysis: {e}")
    
    # Performance analysis
    print("\n6. Performance Analysis...")
    try:
        performance_analyzer = PerformanceAnalyzer()
        portfolio_returns = (portfolio.returns * portfolio.weights).sum(axis=1)
        
        metrics = performance_analyzer.calculate_metrics(portfolio_returns)
        
        print("Key Performance Metrics:")
        for key, value in list(metrics.items())[:10]:  # Show first 10 metrics
            if isinstance(value, float):
                if any(keyword in key.lower() for keyword in ['return', 'volatility']):
                    print(f"{key}: {value:.2%}")
                else:
                    print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
                
    except Exception as e:
        print(f"Error in performance analysis: {e}")
    
    # Portfolio summary
    print("\n7. Portfolio Summary...")
    try:
        summary = portfolio.summary()
        print("\nPortfolio Summary:")
        for key, value in summary.items():
            if isinstance(value, dict):
                print(f"\n{key.upper()}:")
                for subkey, subvalue in value.items():
                    print(f"  {subkey}: {subvalue}")
            elif isinstance(value, list):
                print(f"{key}: {', '.join(map(str, value))}")
            elif isinstance(value, float):
                if any(metric in key.lower() for metric in ['return', 'volatility', 'var']):
                    print(f"{key}: {value:.2%}")
                else:
                    print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
                
    except Exception as e:
        print(f"Error generating summary: {e}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
