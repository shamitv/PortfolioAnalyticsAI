"""
Unit tests for Analyzer class, including efficient frontier functionality.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
from unittest.mock import patch, MagicMock
import base64
import io

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from portfolio_analytics.portfolio import Portfolio
from portfolio_analytics.analyzer import Analyzer
from portfolio_analytics.optimization import PortfolioOptimizer


class TestAnalyzer(unittest.TestCase):
    """Test cases for Analyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.symbols = ['AAPL', 'GOOGL', 'MSFT']
        self.weights = [0.4, 0.3, 0.3]
        self.portfolio = Portfolio(self.symbols, self.weights)
        
        # Create realistic sample data
        dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        # Generate correlated returns for more realistic testing
        n_days = len(dates)
        market_returns = np.random.normal(0.0008, 0.015, n_days)
        
        # Create asset-specific returns with market correlation
        asset_returns = {}
        correlations = [0.7, 0.6, 0.8]  # Market correlations for each asset
        for i, symbol in enumerate(self.symbols):
            idiosyncratic = np.random.normal(0, 0.01, n_days)
            asset_returns[symbol] = correlations[i] * market_returns + np.sqrt(1 - correlations[i]**2) * idiosyncratic
        
        self.sample_returns = pd.DataFrame(asset_returns, index=dates)
        self.sample_data = (1 + self.sample_returns).cumprod() * 100  # Convert to price data
        
        # Set up portfolio with data
        self.portfolio.data = self.sample_data
        self.portfolio.returns = self.sample_returns
        
        self.analyzer = Analyzer(self.portfolio)
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        self.assertIsInstance(self.analyzer.portfolio, Portfolio)
        self.assertIsNotNone(self.analyzer.performance_analyzer)
        self.assertIsNotNone(self.analyzer.risk_model)
        self.assertIsNotNone(self.analyzer.visualizer)
        self.assertIsNotNone(self.analyzer.optimizer)
        self.assertIsInstance(self.analyzer.optimizer, PortfolioOptimizer)
        
        # Check storage containers
        self.assertEqual(self.analyzer.metrics, {})
        self.assertEqual(self.analyzer.greeks, {})
        self.assertEqual(self.analyzer.visualizations, {})
    
    def test_efficient_frontier_chart_creation(self):
        """Test efficient frontier chart creation."""
        chart_base64 = self.analyzer._create_efficient_frontier_chart()
        
        # Verify it's a valid base64 string
        self.assertIsInstance(chart_base64, str)
        self.assertGreater(len(chart_base64), 1000)  # Should be substantial
        
        # Verify it's valid base64
        try:
            decoded = base64.b64decode(chart_base64)
            self.assertGreater(len(decoded), 0)
        except Exception as e:
            self.fail(f"Chart is not valid base64: {e}")
    
    def test_efficient_frontier_in_visualizations(self):
        """Test that efficient frontier is included in visualizations."""
        visualizations = self.analyzer.generate_visualizations()
        
        # Check that efficient frontier is included
        self.assertIn('efficient_frontier', visualizations)
        self.assertIsInstance(visualizations['efficient_frontier'], str)
        self.assertGreater(len(visualizations['efficient_frontier']), 1000)
        
        # Check total number of visualizations
        expected_visualizations = [
            'price_history', 'returns_distribution', 'correlation_matrix',
            'portfolio_composition', 'cumulative_returns', 'drawdown_analysis',
            'risk_return_scatter', 'efficient_frontier', 'rolling_metrics',
            'performance_heatmap', 'greek_sensitivity'
        ]
        self.assertEqual(len(visualizations), len(expected_visualizations))
        
        for viz_name in expected_visualizations:
            self.assertIn(viz_name, visualizations)
    
    def test_efficient_frontier_with_optimization_failure(self):
        """Test efficient frontier chart when optimization fails."""
        # Mock the optimizer to raise an exception
        with patch.object(self.analyzer.optimizer, 'calculate_efficient_frontier') as mock_calc:
            mock_calc.side_effect = Exception("Optimization failed")
            
            # Should not raise exception, but handle gracefully
            chart_base64 = self.analyzer._create_efficient_frontier_chart()
            
            # Should still return a valid chart (without the frontier line)
            self.assertIsInstance(chart_base64, str)
            self.assertGreater(len(chart_base64), 1000)
    
    def test_comprehensive_analysis_includes_efficient_frontier(self):
        """Test that comprehensive analysis includes efficient frontier."""
        analysis = self.analyzer.generate_comprehensive_analysis()
        
        # Check structure
        self.assertIn('visualizations', analysis)
        self.assertIn('efficient_frontier', analysis['visualizations'])
        
        # Verify it's properly included
        efficient_frontier_chart = analysis['visualizations']['efficient_frontier']
        self.assertIsInstance(efficient_frontier_chart, str)
        self.assertGreater(len(efficient_frontier_chart), 1000)
    
    def test_llm_export_includes_efficient_frontier(self):
        """Test that LLM export formats include efficient frontier."""
        # Test comprehensive format
        comprehensive = self.analyzer.export_for_llm(output_format="comprehensive")
        self.assertIn('visualizations', comprehensive)
        self.assertIn('efficient_frontier', comprehensive['visualizations'])
        
        # Test summary format
        summary = self.analyzer.export_for_llm(output_format="summary")
        self.assertIn('visualizations', summary)
        self.assertIn('efficient_frontier', summary['visualizations'])
        
        # Test visuals only format
        visuals_only = self.analyzer.export_for_llm(output_format="visuals_only")
        self.assertIn('efficient_frontier', visuals_only)
        
        # Test metrics only format (should not include visualizations)
        metrics_only = self.analyzer.export_for_llm(output_format="metrics_only")
        self.assertNotIn('visualizations', metrics_only)
    
    def test_efficient_frontier_chart_components(self):
        """Test that efficient frontier chart includes all expected components."""
        # This is more of an integration test to ensure the chart has the right elements
        chart_base64 = self.analyzer._create_efficient_frontier_chart()
        
        # Verify the chart was created successfully
        self.assertIsInstance(chart_base64, str)
        self.assertGreater(len(chart_base64), 1000)
        
        # We can't easily test the visual components without rendering,
        # but we can verify the method completes without error
        # and produces reasonable output size
        
        # Decode to verify it's a valid image
        try:
            image_data = base64.b64decode(chart_base64)
            # PNG files start with these bytes
            self.assertTrue(image_data.startswith(b'\x89PNG'))
        except Exception as e:
            self.fail(f"Generated chart is not a valid PNG: {e}")
    
    def test_openai_messages_include_efficient_frontier(self):
        """Test that OpenAI messages include efficient frontier visualization."""
        messages = self.analyzer.generate_openai_messages(
            include_visualizations=True,
            output_format="comprehensive"
        )
        
        # Should have system and user messages
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]['role'], 'system')
        self.assertEqual(messages[1]['role'], 'user')
        
        # User message should have content array with text and images
        user_content = messages[1]['content']
        self.assertIsInstance(user_content, list)
        
        # Should have text content
        text_content = [item for item in user_content if item['type'] == 'text']
        self.assertGreater(len(text_content), 0)
        
        # Should have image content including efficient frontier
        image_content = [item for item in user_content if item['type'] == 'image_url']
        self.assertGreater(len(image_content), 0)
        
        # Verify efficient frontier is mentioned in text
        text = text_content[0]['text']
        self.assertIn('Efficient Frontier', text)
    
    def test_format_analysis_for_llm_includes_efficient_frontier(self):
        """Test that formatted analysis mentions efficient frontier."""
        analysis_data = self.analyzer.export_for_llm(output_format="comprehensive")
        formatted_text = self.analyzer._format_analysis_for_llm(
            analysis_data, include_visualizations=True
        )
        
        # Should mention efficient frontier in the visualization list
        self.assertIn('Efficient Frontier', formatted_text)
        self.assertIn('VISUALIZATIONS INCLUDED', formatted_text)
    
    def test_portfolio_optimizer_integration(self):
        """Test that the portfolio optimizer is properly integrated."""
        # Verify optimizer exists and has required method
        self.assertIsInstance(self.analyzer.optimizer, PortfolioOptimizer)
        self.assertTrue(hasattr(self.analyzer.optimizer, 'calculate_efficient_frontier'))
        
        # Test that the optimizer method can be called
        try:
            returns, vols, sharpes = self.analyzer.optimizer.calculate_efficient_frontier(
                self.portfolio.returns
            )
            self.assertIsInstance(returns, np.ndarray)
            self.assertIsInstance(vols, np.ndarray)
            self.assertIsInstance(sharpes, np.ndarray)
            self.assertGreater(len(returns), 0)
        except Exception as e:
            self.fail(f"Portfolio optimizer integration failed: {e}")
    
    def test_efficient_frontier_with_minimal_data(self):
        """Test efficient frontier with minimal data points."""
        # Create portfolio with minimal data
        minimal_dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
        np.random.seed(42)
        minimal_returns = pd.DataFrame(
            np.random.randn(len(minimal_dates), len(self.symbols)) * 0.01,
            index=minimal_dates,
            columns=self.symbols
        )
        minimal_data = (1 + minimal_returns).cumprod() * 100
        
        minimal_portfolio = Portfolio(self.symbols, self.weights)
        minimal_portfolio.data = minimal_data
        minimal_portfolio.returns = minimal_returns
        
        minimal_analyzer = Analyzer(minimal_portfolio)
        
        # Should handle minimal data gracefully
        chart_base64 = minimal_analyzer._create_efficient_frontier_chart()
        self.assertIsInstance(chart_base64, str)
        self.assertGreater(len(chart_base64), 1000)
    
    def test_efficient_frontier_with_single_asset(self):
        """Test efficient frontier with single asset portfolio."""
        single_asset_portfolio = Portfolio(['AAPL'], [1.0])
        single_returns = self.sample_returns[['AAPL']]
        single_data = self.sample_data[['AAPL']]
        
        single_asset_portfolio.data = single_data
        single_asset_portfolio.returns = single_returns
        
        single_analyzer = Analyzer(single_asset_portfolio)
        
        # Should handle single asset gracefully
        chart_base64 = single_analyzer._create_efficient_frontier_chart()
        self.assertIsInstance(chart_base64, str)
        self.assertGreater(len(chart_base64), 1000)
    
    def test_visualization_error_handling(self):
        """Test that visualization generation handles errors gracefully."""
        # Create analyzer with problematic data
        problematic_portfolio = Portfolio(self.symbols, self.weights)
        problematic_portfolio.data = None  # No data
        problematic_portfolio.returns = None
        
        problematic_analyzer = Analyzer(problematic_portfolio)
        
        # Should raise appropriate error for missing data
        with self.assertRaises(ValueError):
            problematic_analyzer.generate_comprehensive_analysis()
    
    def test_efficient_frontier_performance(self):
        """Test that efficient frontier generation completes in reasonable time."""
        import time
        
        start_time = time.time()
        chart_base64 = self.analyzer._create_efficient_frontier_chart()
        end_time = time.time()
        
        # Should complete within 10 seconds (generous for CI environments)
        self.assertLess(end_time - start_time, 10.0)
        self.assertIsInstance(chart_base64, str)
        self.assertGreater(len(chart_base64), 1000)


class TestAnalyzerEfficientFrontierEdgeCases(unittest.TestCase):
    """Test edge cases for efficient frontier functionality."""
    
    def setUp(self):
        """Set up edge case test fixtures."""
        self.symbols = ['AAPL', 'GOOGL']
        self.weights = [0.6, 0.4]
        
    def test_highly_correlated_assets(self):
        """Test efficient frontier with highly correlated assets."""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        # Create highly correlated returns
        base_returns = np.random.normal(0.001, 0.02, len(dates))
        correlated_returns = pd.DataFrame({
            'AAPL': base_returns + np.random.normal(0, 0.001, len(dates)),
            'GOOGL': base_returns + np.random.normal(0, 0.001, len(dates))
        }, index=dates)
        
        portfolio = Portfolio(self.symbols, self.weights)
        portfolio.data = (1 + correlated_returns).cumprod() * 100
        portfolio.returns = correlated_returns
        
        analyzer = Analyzer(portfolio)
        
        # Should handle highly correlated assets
        chart_base64 = analyzer._create_efficient_frontier_chart()
        self.assertIsInstance(chart_base64, str)
        self.assertGreater(len(chart_base64), 1000)
    
    def test_negative_returns_assets(self):
        """Test efficient frontier with assets having negative expected returns."""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        # Create returns with negative expectations
        negative_returns = pd.DataFrame({
            'AAPL': np.random.normal(-0.001, 0.02, len(dates)),
            'GOOGL': np.random.normal(-0.0005, 0.018, len(dates))
        }, index=dates)
        
        portfolio = Portfolio(self.symbols, self.weights)
        portfolio.data = (1 + negative_returns).cumprod() * 100
        portfolio.returns = negative_returns
        
        analyzer = Analyzer(portfolio)
        
        # Should handle negative expected returns
        chart_base64 = analyzer._create_efficient_frontier_chart()
        self.assertIsInstance(chart_base64, str)
        self.assertGreater(len(chart_base64), 1000)
    
    def test_high_volatility_assets(self):
        """Test efficient frontier with high volatility assets."""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        # Create high volatility returns
        high_vol_returns = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.1, len(dates)),  # 10% daily vol
            'GOOGL': np.random.normal(0.0008, 0.08, len(dates))  # 8% daily vol
        }, index=dates)
        
        portfolio = Portfolio(self.symbols, self.weights)
        portfolio.data = (1 + high_vol_returns).cumprod() * 100
        portfolio.returns = high_vol_returns
        
        analyzer = Analyzer(portfolio)
        
        # Should handle high volatility assets
        chart_base64 = analyzer._create_efficient_frontier_chart()
        self.assertIsInstance(chart_base64, str)
        self.assertGreater(len(chart_base64), 1000)


if __name__ == '__main__':
    unittest.main()
