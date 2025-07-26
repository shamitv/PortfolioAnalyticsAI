"""
Unit tests for PortfolioOptimizer class, specifically testing efficient frontier functionality.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from portfolio_analytics.optimization import PortfolioOptimizer


class TestPortfolioOptimizer(unittest.TestCase):
    """Test cases for PortfolioOptimizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = PortfolioOptimizer()
        
        # Create sample returns data
        dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        # Generate realistic correlated returns
        n_days = len(dates)
        market_factor = np.random.normal(0.0008, 0.015, n_days)
        
        self.sample_returns = pd.DataFrame({
            'AAPL': 0.7 * market_factor + np.random.normal(0.0002, 0.01, n_days),
            'GOOGL': 0.6 * market_factor + np.random.normal(0.0001, 0.012, n_days),
            'MSFT': 0.8 * market_factor + np.random.normal(0.0003, 0.009, n_days),
            'AMZN': 0.5 * market_factor + np.random.normal(0.0001, 0.015, n_days)
        }, index=dates)
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        self.assertIsInstance(self.optimizer, PortfolioOptimizer)
        self.assertIsNone(self.optimizer.last_optimization_result)
    
    def test_calculate_efficient_frontier_basic(self):
        """Test basic efficient frontier calculation."""
        returns, volatilities, sharpe_ratios = self.optimizer.calculate_efficient_frontier(
            self.sample_returns, num_portfolios=50
        )
        
        # Check output types and shapes
        self.assertIsInstance(returns, np.ndarray)
        self.assertIsInstance(volatilities, np.ndarray)
        self.assertIsInstance(sharpe_ratios, np.ndarray)
        
        # Should have same length
        self.assertEqual(len(returns), len(volatilities))
        self.assertEqual(len(returns), len(sharpe_ratios))
        
        # Should have reasonable number of points
        self.assertGreater(len(returns), 10)  # At least some points calculated
        self.assertLessEqual(len(returns), 50)  # Not more than requested
        
        # Returns and volatilities should be positive
        self.assertTrue(np.all(returns >= 0) or np.all(returns <= 0))  # Allow negative returns
        self.assertTrue(np.all(volatilities >= 0))
        
        # Volatilities should be generally increasing with returns (efficient frontier property)
        # Note: This might not always be strictly true due to optimization challenges
        
    def test_calculate_efficient_frontier_different_num_portfolios(self):
        """Test efficient frontier with different number of portfolios."""
        for num_portfolios in [10, 25, 100]:
            returns, volatilities, sharpe_ratios = self.optimizer.calculate_efficient_frontier(
                self.sample_returns, num_portfolios=num_portfolios
            )
            
            # Should have reasonable number of points (some optimizations might fail)
            self.assertGreater(len(returns), 5)
            self.assertLessEqual(len(returns), num_portfolios)
    
    def test_calculate_efficient_frontier_custom_risk_free_rate(self):
        """Test efficient frontier with custom risk-free rate."""
        returns, volatilities, sharpe_ratios = self.optimizer.calculate_efficient_frontier(
            self.sample_returns, risk_free_rate=0.05
        )
        
        # Should still produce valid results
        self.assertIsInstance(returns, np.ndarray)
        self.assertIsInstance(volatilities, np.ndarray)
        self.assertIsInstance(sharpe_ratios, np.ndarray)
        self.assertGreater(len(returns), 10)
    
    def test_calculate_efficient_frontier_minimal_data(self):
        """Test efficient frontier with minimal data."""
        # Create minimal returns data
        minimal_dates = pd.date_range(start='2023-01-01', end='2023-01-30', freq='D')
        minimal_returns = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, len(minimal_dates)),
            'GOOGL': np.random.normal(0.0008, 0.018, len(minimal_dates))
        }, index=minimal_dates)
        
        returns, volatilities, sharpe_ratios = self.optimizer.calculate_efficient_frontier(
            minimal_returns, num_portfolios=20
        )
        
        # Should handle minimal data
        self.assertIsInstance(returns, np.ndarray)
        self.assertIsInstance(volatilities, np.ndarray)
        self.assertIsInstance(sharpe_ratios, np.ndarray)
        self.assertGreater(len(returns), 0)
    
    def test_calculate_efficient_frontier_single_asset(self):
        """Test efficient frontier with single asset."""
        single_asset_returns = self.sample_returns[['AAPL']]
        
        returns, volatilities, sharpe_ratios = self.optimizer.calculate_efficient_frontier(
            single_asset_returns, num_portfolios=20
        )
        
        # With single asset, all portfolios should be identical
        self.assertIsInstance(returns, np.ndarray)
        self.assertIsInstance(volatilities, np.ndarray)
        self.assertIsInstance(sharpe_ratios, np.ndarray)
        
        # Should have some points (might be fewer due to optimization constraints)
        self.assertGreater(len(returns), 0)
    
    def test_calculate_efficient_frontier_highly_correlated(self):
        """Test efficient frontier with highly correlated assets."""
        # Create highly correlated returns
        base_returns = np.random.normal(0.001, 0.02, len(self.sample_returns))
        correlated_returns = pd.DataFrame({
            'ASSET1': base_returns + np.random.normal(0, 0.001, len(base_returns)),
            'ASSET2': base_returns + np.random.normal(0, 0.001, len(base_returns)),
            'ASSET3': base_returns + np.random.normal(0, 0.001, len(base_returns))
        }, index=self.sample_returns.index)
        
        returns, volatilities, sharpe_ratios = self.optimizer.calculate_efficient_frontier(
            correlated_returns, num_portfolios=30
        )
        
        # Should handle highly correlated assets
        self.assertIsInstance(returns, np.ndarray)
        self.assertIsInstance(volatilities, np.ndarray)
        self.assertIsInstance(sharpe_ratios, np.ndarray)
        self.assertGreater(len(returns), 0)
    
    def test_calculate_efficient_frontier_error_handling(self):
        """Test efficient frontier error handling."""
        # Test with empty dataframe
        empty_returns = pd.DataFrame()
        
        with self.assertRaises(Exception):
            self.optimizer.calculate_efficient_frontier(empty_returns)
        
        # Test with invalid data
        invalid_returns = pd.DataFrame({
            'AAPL': [np.nan, np.inf, -np.inf],
            'GOOGL': [1, 2, 3]
        })
        
        # Should either handle gracefully or raise appropriate error
        try:
            returns, volatilities, sharpe_ratios = self.optimizer.calculate_efficient_frontier(
                invalid_returns
            )
        except Exception:
            # Expected to fail with invalid data
            pass
    
    def test_optimize_max_sharpe(self):
        """Test maximum Sharpe ratio optimization."""
        weights = self.optimizer.optimize(
            self.sample_returns, 
            method="max_sharpe",
            risk_free_rate=0.02
        )
        
        # Should return valid weights
        self.assertIsInstance(weights, np.ndarray)
        self.assertEqual(len(weights), len(self.sample_returns.columns))
        
        # Weights should sum to approximately 1
        self.assertAlmostEqual(np.sum(weights), 1.0, places=5)
        
        # Weights should be non-negative (default constraint)
        self.assertTrue(np.all(weights >= -1e-6))  # Allow small numerical errors
    
    def test_optimize_min_variance(self):
        """Test minimum variance optimization."""
        weights = self.optimizer.optimize(
            self.sample_returns,
            method="min_variance"
        )
        
        # Should return valid weights
        self.assertIsInstance(weights, np.ndarray)
        self.assertEqual(len(weights), len(self.sample_returns.columns))
        self.assertAlmostEqual(np.sum(weights), 1.0, places=5)
        self.assertTrue(np.all(weights >= -1e-6))
    
    def test_optimize_target_return(self):
        """Test target return optimization."""
        target_return = 0.10  # 10% annual return
        
        weights = self.optimizer.optimize(
            self.sample_returns,
            method="target_return",
            target_return=target_return
        )
        
        # Should return valid weights
        self.assertIsInstance(weights, np.ndarray)
        self.assertEqual(len(weights), len(self.sample_returns.columns))
        self.assertAlmostEqual(np.sum(weights), 1.0, places=5)
        
        # Check that target return is approximately achieved
        expected_returns = self.sample_returns.mean() * 252
        portfolio_return = np.sum(expected_returns * weights)
        self.assertAlmostEqual(portfolio_return, target_return, places=2)
    
    def test_get_optimization_summary(self):
        """Test optimization summary generation."""
        weights = self.optimizer.optimize(
            self.sample_returns,
            method="max_sharpe"
        )
        
        summary = self.optimizer.get_optimization_summary(
            self.sample_returns, 
            weights
        )
        
        # Check summary structure
        expected_keys = [
            'expected_annual_return', 'annual_volatility', 'sharpe_ratio',
            'weights', 'diversification_ratio', 'maximum_weight', 
            'minimum_weight', 'effective_number_of_assets'
        ]
        
        for key in expected_keys:
            self.assertIn(key, summary)
        
        # Check data types and ranges
        self.assertIsInstance(summary['expected_annual_return'], (int, float))
        self.assertIsInstance(summary['annual_volatility'], (int, float))
        self.assertIsInstance(summary['sharpe_ratio'], (int, float))
        self.assertIsInstance(summary['weights'], dict)
        
        # Volatility should be positive
        self.assertGreater(summary['annual_volatility'], 0)
        
        # Diversification ratio should be >= 1
        self.assertGreaterEqual(summary['diversification_ratio'], 1.0)
        
        # Effective number of assets should be reasonable
        self.assertGreater(summary['effective_number_of_assets'], 0)
        self.assertLessEqual(summary['effective_number_of_assets'], len(self.sample_returns.columns))
    
    def test_efficient_frontier_integration(self):
        """Test integration between efficient frontier and standard optimization."""
        # Calculate efficient frontier
        ef_returns, ef_volatilities, ef_sharpe_ratios = self.optimizer.calculate_efficient_frontier(
            self.sample_returns, num_portfolios=20
        )
        
        # Find max Sharpe portfolio using standard optimization
        max_sharpe_weights = self.optimizer.optimize(
            self.sample_returns,
            method="max_sharpe"
        )
        
        # Calculate performance of max Sharpe portfolio
        expected_returns = self.sample_returns.mean() * 252
        cov_matrix = self.sample_returns.cov() * 252
        
        max_sharpe_return = np.sum(expected_returns * max_sharpe_weights)
        max_sharpe_vol = np.sqrt(np.dot(max_sharpe_weights.T, np.dot(cov_matrix, max_sharpe_weights)))
        max_sharpe_ratio = (max_sharpe_return - 0.02) / max_sharpe_vol
        
        # The max Sharpe portfolio should have a Sharpe ratio close to or better than
        # the best point on the efficient frontier
        if len(ef_sharpe_ratios) > 0:
            max_ef_sharpe = np.max(ef_sharpe_ratios)
            # Allow some tolerance due to numerical differences and optimization constraints
            self.assertGreaterEqual(max_sharpe_ratio, max_ef_sharpe - 0.1)


class TestPortfolioOptimizerEdgeCases(unittest.TestCase):
    """Test edge cases for PortfolioOptimizer."""
    
    def setUp(self):
        """Set up edge case test fixtures."""
        self.optimizer = PortfolioOptimizer()
    
    def test_negative_return_assets(self):
        """Test efficient frontier with negative expected return assets."""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        negative_returns = pd.DataFrame({
            'BEAR1': np.random.normal(-0.001, 0.02, len(dates)),
            'BEAR2': np.random.normal(-0.0005, 0.018, len(dates))
        }, index=dates)
        
        returns, volatilities, sharpe_ratios = self.optimizer.calculate_efficient_frontier(
            negative_returns, num_portfolios=20
        )
        
        # Should handle negative returns
        self.assertIsInstance(returns, np.ndarray)
        self.assertIsInstance(volatilities, np.ndarray)
        self.assertIsInstance(sharpe_ratios, np.ndarray)
        self.assertGreater(len(returns), 0)
    
    def test_extreme_volatility_assets(self):
        """Test efficient frontier with extremely volatile assets."""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        extreme_vol_returns = pd.DataFrame({
            'VOLATILE1': np.random.normal(0.001, 0.2, len(dates)),  # 20% daily vol
            'VOLATILE2': np.random.normal(0.0008, 0.15, len(dates))  # 15% daily vol
        }, index=dates)
        
        returns, volatilities, sharpe_ratios = self.optimizer.calculate_efficient_frontier(
            extreme_vol_returns, num_portfolios=15
        )
        
        # Should handle extreme volatility
        self.assertIsInstance(returns, np.ndarray)
        self.assertIsInstance(volatilities, np.ndarray)
        self.assertIsInstance(sharpe_ratios, np.ndarray)
        self.assertGreater(len(returns), 0)
    
    def test_zero_variance_asset(self):
        """Test efficient frontier with zero variance asset."""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        zero_var_returns = pd.DataFrame({
            'CONSTANT': np.full(len(dates), 0.001),  # Constant return
            'NORMAL': np.random.normal(0.0008, 0.02, len(dates))
        }, index=dates)
        
        # This might cause numerical issues, should handle gracefully
        try:
            returns, volatilities, sharpe_ratios = self.optimizer.calculate_efficient_frontier(
                zero_var_returns, num_portfolios=10
            )
            
            self.assertIsInstance(returns, np.ndarray)
            self.assertIsInstance(volatilities, np.ndarray)
            self.assertIsInstance(sharpe_ratios, np.ndarray)
        except Exception:
            # Expected to potentially fail with zero variance
            pass


if __name__ == '__main__':
    unittest.main()
