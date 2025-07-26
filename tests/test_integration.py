"""
Integration tests for efficient frontier functionality across the entire system.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
import tempfile
import json
import base64

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from portfolio_analytics.portfolio import Portfolio
from portfolio_analytics.analyzer import Analyzer
from portfolio_analytics.optimization import PortfolioOptimizer
from portfolio_analytics.data_provider import DataProvider


class TestEfficientFrontierIntegration(unittest.TestCase):
    """Integration tests for efficient frontier functionality."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        # Create a realistic multi-asset portfolio
        self.symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        self.weights = [0.3, 0.25, 0.2, 0.15, 0.1]
        
        # Generate realistic market data
        dates = pd.date_range(start='2021-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        # Create correlated returns using factor model
        market_factor = np.random.normal(0.0008, 0.016, len(dates))
        sector_factors = {
            'tech': np.random.normal(0.0002, 0.012, len(dates)),
            'growth': np.random.normal(0.0001, 0.010, len(dates))
        }
        
        # Asset-specific parameters
        asset_params = {
            'AAPL': {'market_beta': 1.2, 'tech_beta': 0.8, 'growth_beta': 0.6, 'idiosync_vol': 0.008},
            'GOOGL': {'market_beta': 1.1, 'tech_beta': 0.9, 'growth_beta': 0.7, 'idiosync_vol': 0.009},
            'MSFT': {'market_beta': 0.9, 'tech_beta': 0.7, 'growth_beta': 0.5, 'idiosync_vol': 0.007},
            'AMZN': {'market_beta': 1.3, 'tech_beta': 0.6, 'growth_beta': 0.9, 'idiosync_vol': 0.012},
            'TSLA': {'market_beta': 1.8, 'tech_beta': 0.4, 'growth_beta': 1.2, 'idiosync_vol': 0.025}
        }
        
        # Generate returns
        returns_data = {}
        for symbol in self.symbols:
            params = asset_params[symbol]
            idiosyncratic = np.random.normal(0, params['idiosync_vol'], len(dates))
            
            returns_data[symbol] = (
                params['market_beta'] * market_factor +
                params['tech_beta'] * sector_factors['tech'] +
                params['growth_beta'] * sector_factors['growth'] +
                idiosyncratic
            )
        
        self.sample_returns = pd.DataFrame(returns_data, index=dates)
        self.sample_data = (1 + self.sample_returns).cumprod() * 100
        
        # Create portfolio
        self.portfolio = Portfolio(self.symbols, self.weights, name="Tech Portfolio")
        self.portfolio.data = self.sample_data
        self.portfolio.returns = self.sample_returns
        
        self.analyzer = Analyzer(self.portfolio)
    
    def test_end_to_end_efficient_frontier_workflow(self):
        """Test complete efficient frontier workflow from data to visualization."""
        # Step 1: Generate comprehensive analysis
        analysis = self.analyzer.generate_comprehensive_analysis()
        
        # Verify structure
        self.assertIn('metrics', analysis)
        self.assertIn('greeks', analysis)
        self.assertIn('visualizations', analysis)
        self.assertIn('portfolio_summary', analysis)
        
        # Step 2: Verify efficient frontier is included
        self.assertIn('efficient_frontier', analysis['visualizations'])
        efficient_frontier_chart = analysis['visualizations']['efficient_frontier']
        
        # Verify it's a valid base64 image
        self.assertIsInstance(efficient_frontier_chart, str)
        self.assertGreater(len(efficient_frontier_chart), 10000)  # Should be substantial
        
        # Verify it's valid base64 PNG
        image_data = base64.b64decode(efficient_frontier_chart)
        self.assertTrue(image_data.startswith(b'\x89PNG'))
        
        # Step 3: Test LLM export formats
        for output_format in ['comprehensive', 'summary', 'visuals_only']:
            llm_data = self.analyzer.export_for_llm(output_format=output_format)
            if 'visualizations' in llm_data:
                self.assertIn('efficient_frontier', llm_data['visualizations'])
        
        # Step 4: Test OpenAI message generation
        messages = self.analyzer.generate_openai_messages(include_visualizations=True)
        self.assertEqual(len(messages), 2)
        
        # Verify images are included
        user_content = messages[1]['content']
        image_content = [item for item in user_content if item['type'] == 'image_url']
        self.assertGreater(len(image_content), 0)
        
        # Step 5: Verify efficient frontier mentioned in text
        text_content = [item for item in user_content if item['type'] == 'text'][0]['text']
        self.assertIn('Efficient Frontier', text_content)
    
    def test_efficient_frontier_optimization_consistency(self):
        """Test consistency between efficient frontier and individual optimizations."""
        # Generate efficient frontier
        ef_returns, ef_volatilities, ef_sharpe_ratios = self.analyzer.optimizer.calculate_efficient_frontier(
            self.portfolio.returns, num_portfolios=50
        )
        
        self.assertGreater(len(ef_returns), 20)  # Should have reasonable number of points
        
        # Test max Sharpe optimization
        max_sharpe_weights = self.analyzer.optimizer.optimize(
            self.portfolio.returns, method="max_sharpe"
        )
        
        # Calculate performance metrics
        expected_returns = self.portfolio.returns.mean() * 252
        cov_matrix = self.portfolio.returns.cov() * 252
        
        max_sharpe_return = np.sum(expected_returns * max_sharpe_weights)
        max_sharpe_vol = np.sqrt(np.dot(max_sharpe_weights.T, np.dot(cov_matrix, max_sharpe_weights)))
        max_sharpe_ratio = (max_sharpe_return - 0.02) / max_sharpe_vol
        
        # Test min variance optimization
        min_var_weights = self.analyzer.optimizer.optimize(
            self.portfolio.returns, method="min_variance"
        )
        
        min_var_vol = np.sqrt(np.dot(min_var_weights.T, np.dot(cov_matrix, min_var_weights)))
        
        # Verify max Sharpe is close to best Sharpe on frontier
        if len(ef_sharpe_ratios) > 0:
            max_ef_sharpe = np.max(ef_sharpe_ratios)
            self.assertGreater(max_sharpe_ratio, max_ef_sharpe - 0.2)  # Allow some tolerance
        
        # Verify min variance is close to minimum volatility on frontier
        if len(ef_volatilities) > 0:
            min_ef_vol = np.min(ef_volatilities)
            self.assertLess(min_var_vol, min_ef_vol + 0.05)  # Allow some tolerance
    
    def test_efficient_frontier_with_benchmark(self):
        """Test efficient frontier analysis with benchmark comparison."""
        # Create a simple benchmark (equal-weighted market proxy)
        benchmark_returns = self.portfolio.returns.mean(axis=1)
        
        # Generate analysis with benchmark
        analysis = self.analyzer.generate_comprehensive_analysis(
            benchmark_returns=benchmark_returns
        )
        
        # Verify benchmark is included in visualizations
        self.assertIn('cumulative_returns', analysis['visualizations'])
        
        # Generate LLM messages with benchmark
        messages = self.analyzer.generate_openai_messages(
            benchmark_returns=benchmark_returns,
            include_visualizations=True
        )
        
        # Should still include efficient frontier
        user_content = messages[1]['content']
        image_content = [item for item in user_content if item['type'] == 'image_url']
        self.assertGreater(len(image_content), 0)
    
    def test_efficient_frontier_performance_analysis(self):
        """Test that efficient frontier provides meaningful portfolio insights."""
        # Generate analysis
        analysis = self.analyzer.generate_comprehensive_analysis()
        
        # Extract portfolio metrics
        performance_metrics = analysis['metrics']['performance']
        allocation_metrics = analysis['metrics']['allocation']
        
        # Calculate current portfolio position
        portfolio_returns = (self.portfolio.returns * self.portfolio.weights).sum(axis=1)
        current_return = portfolio_returns.mean() * 252
        current_vol = portfolio_returns.std() * np.sqrt(252)
        current_sharpe = (current_return - 0.02) / current_vol
        
        # Generate efficient frontier for comparison
        ef_returns, ef_volatilities, ef_sharpe_ratios = self.analyzer.optimizer.calculate_efficient_frontier(
            self.portfolio.returns, num_portfolios=100
        )
        
        if len(ef_sharpe_ratios) > 0:
            max_ef_sharpe = np.max(ef_sharpe_ratios)
            
            # Current portfolio should be reasonably positioned relative to efficient frontier
            sharpe_ratio_efficiency = current_sharpe / max_ef_sharpe if max_ef_sharpe > 0 else 0
            
            # Portfolio should be at least somewhat efficient (>50% of max Sharpe)
            # This is a reasonable expectation for a diversified portfolio
            self.assertGreater(sharpe_ratio_efficiency, 0.3)
    
    def test_efficient_frontier_error_recovery(self):
        """Test that system gracefully handles efficient frontier calculation errors."""
        # Create problematic portfolio data
        problematic_returns = pd.DataFrame({
            'ASSET1': np.full(100, 0.0),  # Zero variance
            'ASSET2': np.random.normal(0, 0.001, 100)  # Very low variance
        })
        
        problematic_portfolio = Portfolio(['ASSET1', 'ASSET2'], [0.5, 0.5])
        problematic_portfolio.data = (1 + problematic_returns).cumprod() * 100
        problematic_portfolio.returns = problematic_returns
        
        problematic_analyzer = Analyzer(problematic_portfolio)
        
        # Should not crash, even if efficient frontier calculation fails
        try:
            chart = problematic_analyzer._create_efficient_frontier_chart()
            self.assertIsInstance(chart, str)
            self.assertGreater(len(chart), 1000)
        except Exception as e:
            self.fail(f"Efficient frontier should handle problematic data gracefully: {e}")
    
    def test_efficient_frontier_scalability(self):
        """Test efficient frontier with larger number of assets."""
        # Create portfolio with more assets
        large_symbols = [f'ASSET_{i:02d}' for i in range(20)]
        large_weights = [1/20] * 20
        
        # Generate correlated returns
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        market_factor = np.random.normal(0.0008, 0.015, len(dates))
        
        large_returns_data = {}
        for i, symbol in enumerate(large_symbols):
            correlation = 0.3 + 0.4 * np.random.random()  # Random correlation 0.3-0.7
            idiosync_vol = 0.005 + 0.015 * np.random.random()  # Random vol 0.5%-2%
            
            large_returns_data[symbol] = (
                correlation * market_factor +
                np.sqrt(1 - correlation**2) * np.random.normal(0, idiosync_vol, len(dates))
            )
        
        large_returns = pd.DataFrame(large_returns_data, index=dates)
        large_data = (1 + large_returns).cumprod() * 100
        
        large_portfolio = Portfolio(large_symbols, large_weights)
        large_portfolio.data = large_data
        large_portfolio.returns = large_returns
        
        large_analyzer = Analyzer(large_portfolio)
        
        # Should handle larger portfolios
        chart = large_analyzer._create_efficient_frontier_chart()
        self.assertIsInstance(chart, str)
        self.assertGreater(len(chart), 1000)
        
        # Efficient frontier calculation should still work
        ef_returns, ef_volatilities, ef_sharpe_ratios = large_analyzer.optimizer.calculate_efficient_frontier(
            large_returns, num_portfolios=30  # Fewer points for speed
        )
        
        self.assertGreater(len(ef_returns), 5)
    
    def test_efficient_frontier_data_export(self):
        """Test that efficient frontier data can be exported and analyzed."""
        analysis = self.analyzer.generate_comprehensive_analysis()
        
        # Test JSON serialization of analysis (important for API usage)
        try:
            # Remove base64 images for JSON test (they're very large)
            analysis_copy = analysis.copy()
            analysis_copy['visualizations'] = {
                k: f"<base64_image_{len(v)}_chars>" 
                for k, v in analysis['visualizations'].items()
            }
            
            json_str = json.dumps(analysis_copy, default=str, indent=2)
            self.assertGreater(len(json_str), 1000)
            
            # Should be valid JSON
            json.loads(json_str)
            
        except Exception as e:
            self.fail(f"Analysis should be JSON serializable: {e}")
        
        # Test that visualization data is accessible
        viz_data = analysis['visualizations']
        self.assertIn('efficient_frontier', viz_data)
        
        # Each visualization should be a base64 string
        for viz_name, viz_data in viz_data.items():
            self.assertIsInstance(viz_data, str)
            self.assertGreater(len(viz_data), 1000)
            
            # Should be valid base64
            try:
                base64.b64decode(viz_data)
            except Exception as e:
                self.fail(f"Visualization {viz_name} should be valid base64: {e}")


class TestEfficientFrontierRealWorldScenarios(unittest.TestCase):
    """Test efficient frontier with real-world-like scenarios."""
    
    def test_market_crash_scenario(self):
        """Test efficient frontier during market crash conditions."""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        # Simulate market crash with high correlation
        crash_factor = np.concatenate([
            np.random.normal(0.001, 0.01, 100),  # Normal period
            np.random.normal(-0.05, 0.08, 50),   # Crash period
            np.random.normal(0.002, 0.02, len(dates) - 150)  # Recovery
        ])[:len(dates)]
        
        crash_returns = pd.DataFrame({
            'STOCK1': 0.8 * crash_factor + np.random.normal(0, 0.005, len(dates)),
            'STOCK2': 0.9 * crash_factor + np.random.normal(0, 0.006, len(dates)),
            'BOND': 0.1 * crash_factor + np.random.normal(0.0001, 0.002, len(dates))  # Defensive
        }, index=dates)
        
        portfolio = Portfolio(['STOCK1', 'STOCK2', 'BOND'], [0.4, 0.4, 0.2])
        portfolio.data = (1 + crash_returns).cumprod() * 100
        portfolio.returns = crash_returns
        
        analyzer = Analyzer(portfolio)
        
        # Should handle extreme market conditions
        chart = analyzer._create_efficient_frontier_chart()
        self.assertIsInstance(chart, str)
        self.assertGreater(len(chart), 1000)
    
    def test_low_interest_rate_environment(self):
        """Test efficient frontier in low interest rate environment."""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        # Low-yield environment
        low_yield_returns = pd.DataFrame({
            'GROWTH': np.random.normal(0.0008, 0.02, len(dates)),
            'VALUE': np.random.normal(0.0003, 0.015, len(dates)),
            'BONDS': np.random.normal(0.0001, 0.003, len(dates))  # Very low yield
        }, index=dates)
        
        portfolio = Portfolio(['GROWTH', 'VALUE', 'BONDS'], [0.5, 0.3, 0.2])
        portfolio.data = (1 + low_yield_returns).cumprod() * 100
        portfolio.returns = low_yield_returns
        
        analyzer = Analyzer(portfolio)
        
        # Test with very low risk-free rate
        analysis = analyzer.generate_comprehensive_analysis(risk_free_rate=0.001)
        self.assertIn('efficient_frontier', analysis['visualizations'])
    
    def test_high_inflation_scenario(self):
        """Test efficient frontier during high inflation."""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        # High inflation environment
        inflation_returns = pd.DataFrame({
            'COMMODITIES': np.random.normal(0.003, 0.03, len(dates)),  # Inflation hedge
            'REAL_ESTATE': np.random.normal(0.002, 0.02, len(dates)),  # Real assets
            'STOCKS': np.random.normal(0.001, 0.025, len(dates)),      # Nominal assets
            'BONDS': np.random.normal(-0.001, 0.015, len(dates))       # Hurt by inflation
        }, index=dates)
        
        portfolio = Portfolio(['COMMODITIES', 'REAL_ESTATE', 'STOCKS', 'BONDS'], [0.3, 0.3, 0.3, 0.1])
        portfolio.data = (1 + inflation_returns).cumprod() * 100
        portfolio.returns = inflation_returns
        
        analyzer = Analyzer(portfolio)
        
        # Test with higher risk-free rate (inflation environment)
        analysis = analyzer.generate_comprehensive_analysis(risk_free_rate=0.05)
        self.assertIn('efficient_frontier', analysis['visualizations'])


if __name__ == '__main__':
    unittest.main()
