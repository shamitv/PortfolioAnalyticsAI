"""
Unit tests for Portfolio class.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from portfolio_analytics.portfolio import Portfolio
from portfolio_analytics.data_provider import DataProvider


class TestPortfolio(unittest.TestCase):
    """Test cases for Portfolio class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.symbols = ['AAPL', 'GOOGL', 'MSFT']
        self.weights = [0.4, 0.3, 0.3]
        self.portfolio = Portfolio(self.symbols, self.weights)
        
        # Create sample data
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        self.sample_data = pd.DataFrame(
            data=np.random.randn(len(dates), len(self.symbols)).cumsum(axis=0) + 100,
            index=dates,
            columns=self.symbols
        )
        
        self.sample_returns = self.sample_data.pct_change().dropna()
    
    def test_portfolio_initialization(self):
        """Test portfolio initialization."""
        self.assertEqual(self.portfolio.symbols, self.symbols)
        np.testing.assert_array_almost_equal(self.portfolio.weights, self.weights)
        self.assertEqual(self.portfolio.name, "Portfolio")
        
    def test_portfolio_initialization_equal_weights(self):
        """Test portfolio initialization with equal weights."""
        portfolio = Portfolio(self.symbols)
        expected_weights = [1/3, 1/3, 1/3]
        np.testing.assert_array_almost_equal(portfolio.weights, expected_weights)
    
    def test_portfolio_initialization_invalid_weights(self):
        """Test portfolio initialization with invalid weights."""
        # Wrong number of weights
        with self.assertRaises(ValueError):
            Portfolio(self.symbols, [0.5, 0.5])
        
        # Weights don't sum to 1
        with self.assertRaises(ValueError):
            Portfolio(self.symbols, [0.5, 0.5, 0.6])
    
    def test_portfolio_return_calculation(self):
        """Test portfolio return calculation."""
        # Manually set data for testing
        self.portfolio.data = self.sample_data
        self.portfolio.returns = self.sample_returns
        
        # Calculate expected portfolio return
        weighted_returns = (self.sample_returns * self.weights).sum(axis=1)
        expected_return = weighted_returns.mean()
        
        calculated_return = self.portfolio.portfolio_return()
        self.assertAlmostEqual(calculated_return, expected_return, places=6)
    
    def test_annual_return_calculation(self):
        """Test annual return calculation."""
        self.portfolio.data = self.sample_data
        self.portfolio.returns = self.sample_returns
        
        portfolio_return = self.portfolio.portfolio_return()
        expected_annual_return = portfolio_return * 252
        
        calculated_annual_return = self.portfolio.annual_return()
        self.assertAlmostEqual(calculated_annual_return, expected_annual_return, places=6)
    
    def test_portfolio_volatility_calculation(self):
        """Test portfolio volatility calculation."""
        self.portfolio.data = self.sample_data
        self.portfolio.returns = self.sample_returns
        
        weighted_returns = (self.sample_returns * self.weights).sum(axis=1)
        expected_volatility = weighted_returns.std()
        
        calculated_volatility = self.portfolio.portfolio_volatility()
        self.assertAlmostEqual(calculated_volatility, expected_volatility, places=6)
    
    def test_annual_volatility_calculation(self):
        """Test annual volatility calculation."""
        self.portfolio.data = self.sample_data
        self.portfolio.returns = self.sample_returns
        
        portfolio_volatility = self.portfolio.portfolio_volatility()
        expected_annual_volatility = portfolio_volatility * np.sqrt(252)
        
        calculated_annual_volatility = self.portfolio.annual_volatility()
        self.assertAlmostEqual(calculated_annual_volatility, expected_annual_volatility, places=6)
    
    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation."""
        self.portfolio.data = self.sample_data
        self.portfolio.returns = self.sample_returns
        
        risk_free_rate = 0.02
        annual_return = self.portfolio.annual_return()
        annual_volatility = self.portfolio.annual_volatility()
        expected_sharpe = (annual_return - risk_free_rate) / annual_volatility
        
        calculated_sharpe = self.portfolio.sharpe_ratio(risk_free_rate)
        self.assertAlmostEqual(calculated_sharpe, expected_sharpe, places=6)
    
    def test_correlation_matrix(self):
        """Test correlation matrix calculation."""
        self.portfolio.data = self.sample_data
        self.portfolio.returns = self.sample_returns
        
        expected_correlation = self.sample_returns.corr()
        calculated_correlation = self.portfolio.get_correlation_matrix()
        
        pd.testing.assert_frame_equal(calculated_correlation, expected_correlation)
    
    def test_portfolio_summary_no_data(self):
        """Test portfolio summary without data."""
        summary = self.portfolio.summary()
        
        self.assertEqual(summary['name'], "Portfolio")
        self.assertEqual(summary['symbols'], self.symbols)
        self.assertEqual(summary['weights'], self.weights)
        self.assertEqual(summary['status'], "No data loaded")
    
    def test_portfolio_summary_with_data(self):
        """Test portfolio summary with data."""
        self.portfolio.data = self.sample_data
        self.portfolio.returns = self.sample_returns
        
        summary = self.portfolio.summary()
        
        self.assertEqual(summary['name'], "Portfolio")
        self.assertEqual(summary['symbols'], self.symbols)
        self.assertEqual(summary['weights'], self.weights)
        self.assertIn('annual_return', summary)
        self.assertIn('annual_volatility', summary)
        self.assertIn('sharpe_ratio', summary)
        self.assertIn('var_95', summary)
        self.assertIn('data_period', summary)
    
    def test_portfolio_repr(self):
        """Test portfolio string representation."""
        expected_repr = f"Portfolio(name='Portfolio', symbols={self.symbols}, n_assets={len(self.symbols)})"
        self.assertEqual(repr(self.portfolio), expected_repr)


class TestPortfolioEdgeCases(unittest.TestCase):
    """Test edge cases for Portfolio class."""
    
    def test_empty_returns_handling(self):
        """Test handling of empty returns."""
        portfolio = Portfolio(['AAPL'])
        
        # Test methods that should raise ValueError with no data
        with self.assertRaises(ValueError):
            portfolio.portfolio_return()
        
        with self.assertRaises(ValueError):
            portfolio.annual_return()
        
        with self.assertRaises(ValueError):
            portfolio.portfolio_volatility()
        
        with self.assertRaises(ValueError):
            portfolio.annual_volatility()
        
        with self.assertRaises(ValueError):
            portfolio.get_correlation_matrix()
    
    def test_single_asset_portfolio(self):
        """Test portfolio with single asset."""
        portfolio = Portfolio(['AAPL'])
        
        # Create simple test data
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        data = pd.DataFrame(
            data=np.random.randn(100).cumsum() + 100,
            index=dates,
            columns=['AAPL']
        )
        
        portfolio.data = data
        portfolio.returns = data.pct_change().dropna()
        
        # Test calculations work for single asset
        self.assertIsInstance(portfolio.portfolio_return(), float)
        self.assertIsInstance(portfolio.annual_return(), float)
        self.assertIsInstance(portfolio.portfolio_volatility(), float)
        self.assertIsInstance(portfolio.annual_volatility(), float)
        self.assertIsInstance(portfolio.sharpe_ratio(), float)


if __name__ == '__main__':
    unittest.main()
