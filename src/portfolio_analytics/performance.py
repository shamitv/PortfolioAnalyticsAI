"""
Performance analysis and metrics calculation.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from datetime import datetime


class PerformanceAnalyzer:
    """
    Portfolio performance analysis and metrics calculation.
    
    Provides comprehensive performance metrics including returns,
    risk-adjusted returns, and benchmark comparisons.
    """
    
    def __init__(self):
        """Initialize performance analyzer."""
        pass
    
    def calculate_metrics(self, 
                         returns: pd.Series,
                         benchmark_returns: Optional[pd.Series] = None,
                         risk_free_rate: float = 0.02) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            returns: Portfolio returns series
            benchmark_returns: Benchmark returns (optional)
            risk_free_rate: Risk-free rate (annualized)
        
        Returns:
            Dictionary of performance metrics
        """
        metrics = {}
        
        # Basic return metrics
        metrics.update(self._calculate_return_metrics(returns))
        
        # Risk metrics
        metrics.update(self._calculate_risk_metrics(returns))
        
        # Risk-adjusted metrics
        metrics.update(self._calculate_risk_adjusted_metrics(returns, risk_free_rate))
        
        # Benchmark comparison (if provided)
        if benchmark_returns is not None:
            metrics.update(self._calculate_benchmark_metrics(returns, benchmark_returns, risk_free_rate))
        
        return metrics
    
    def _calculate_return_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate basic return metrics."""
        if len(returns) == 0:
            return {}
        
        total_return = (1 + returns).prod() - 1
        annual_return = returns.mean() * 252
        
        # Calculate monthly and quarterly returns
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        quarterly_returns = returns.resample('Q').apply(lambda x: (1 + x).prod() - 1)
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'average_monthly_return': monthly_returns.mean(),
            'average_quarterly_return': quarterly_returns.mean(),
            'best_month': monthly_returns.max() if len(monthly_returns) > 0 else 0,
            'worst_month': monthly_returns.min() if len(monthly_returns) > 0 else 0,
            'positive_months': (monthly_returns > 0).sum() if len(monthly_returns) > 0 else 0,
            'negative_months': (monthly_returns < 0).sum() if len(monthly_returns) > 0 else 0,
        }
    
    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk metrics."""
        if len(returns) == 0:
            return {}
        
        annual_volatility = returns.std() * np.sqrt(252)
        
        return {
            'annual_volatility': annual_volatility,
            'daily_volatility': returns.std(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
        }
    
    def _calculate_risk_adjusted_metrics(self, 
                                       returns: pd.Series,
                                       risk_free_rate: float) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics."""
        if len(returns) == 0:
            return {}
        
        annual_return = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility != 0 else 0
        
        # Calmar ratio (using simplified max drawdown calculation)
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        calmar_ratio = annual_return / max_drawdown if max_drawdown != 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annual_return - risk_free_rate) / downside_deviation if downside_deviation != 0 else 0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
        }
    
    def _calculate_benchmark_metrics(self, 
                                   returns: pd.Series,
                                   benchmark_returns: pd.Series,
                                   risk_free_rate: float) -> Dict[str, float]:
        """Calculate benchmark comparison metrics."""
        # Align the series
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
        
        if len(aligned_returns) == 0:
            return {}
        
        # Active returns
        active_returns = aligned_returns - aligned_benchmark
        
        # Alpha and Beta calculation
        try:
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(aligned_benchmark, aligned_returns)
            
            beta = slope
            alpha = intercept * 252  # Annualized alpha
            r_squared = r_value ** 2
            
        except ImportError:
            # Fallback calculation without scipy
            covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
            benchmark_variance = np.var(aligned_benchmark)
            beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
            alpha = (aligned_returns.mean() - beta * aligned_benchmark.mean()) * 252
            r_squared = np.corrcoef(aligned_returns, aligned_benchmark)[0, 1] ** 2
        
        # Tracking error
        tracking_error = active_returns.std() * np.sqrt(252)
        
        # Information ratio
        information_ratio = (active_returns.mean() * 252) / tracking_error if tracking_error != 0 else 0
        
        # Up/Down capture ratios
        up_market = aligned_benchmark > 0
        down_market = aligned_benchmark < 0
        
        up_capture = 0
        down_capture = 0
        
        if up_market.any():
            portfolio_up_return = aligned_returns[up_market].mean()
            benchmark_up_return = aligned_benchmark[up_market].mean()
            up_capture = portfolio_up_return / benchmark_up_return if benchmark_up_return != 0 else 0
        
        if down_market.any():
            portfolio_down_return = aligned_returns[down_market].mean()
            benchmark_down_return = aligned_benchmark[down_market].mean()
            down_capture = portfolio_down_return / benchmark_down_return if benchmark_down_return != 0 else 0
        
        return {
            'alpha': alpha,
            'beta': beta,
            'r_squared': r_squared,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'up_capture_ratio': up_capture,
            'down_capture_ratio': down_capture,
            'correlation_with_benchmark': np.corrcoef(aligned_returns, aligned_benchmark)[0, 1],
        }
    
    def calculate_rolling_metrics(self, 
                                returns: pd.Series,
                                window: int = 252,
                                metrics: List[str] = ['return', 'volatility', 'sharpe']) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.
        
        Args:
            returns: Portfolio returns series
            window: Rolling window size (default: 1 year = 252 days)
            metrics: List of metrics to calculate
        
        Returns:
            DataFrame with rolling metrics
        """
        rolling_metrics = pd.DataFrame(index=returns.index)
        
        if 'return' in metrics:
            rolling_metrics['rolling_return'] = returns.rolling(window=window).mean() * 252
        
        if 'volatility' in metrics:
            rolling_metrics['rolling_volatility'] = returns.rolling(window=window).std() * np.sqrt(252)
        
        if 'sharpe' in metrics:
            rolling_return = returns.rolling(window=window).mean() * 252
            rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
            rolling_metrics['rolling_sharpe'] = (rolling_return - 0.02) / rolling_vol  # Assuming 2% risk-free rate
        
        if 'max_drawdown' in metrics:
            def rolling_max_drawdown(series):
                cumulative = (1 + series).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                return abs(drawdown.min())
            
            rolling_metrics['rolling_max_drawdown'] = returns.rolling(window=window).apply(rolling_max_drawdown)
        
        return rolling_metrics.dropna()
    
    def performance_attribution(self, 
                              returns: pd.DataFrame,
                              weights: np.ndarray,
                              benchmark_weights: Optional[np.ndarray] = None) -> Dict[str, any]:
        """
        Perform basic performance attribution analysis.
        
        Args:
            returns: DataFrame of asset returns
            weights: Portfolio weights
            benchmark_weights: Benchmark weights (optional)
        
        Returns:
            Dictionary with attribution results
        """
        # Portfolio return
        portfolio_return = (returns * weights).sum(axis=1).mean() * 252
        
        # Asset contributions
        asset_contributions = {}
        for i, asset in enumerate(returns.columns):
            contribution = returns[asset].mean() * weights[i] * 252
            asset_contributions[asset] = contribution
        
        result = {
            'portfolio_return': portfolio_return,
            'asset_contributions': asset_contributions,
            'total_contribution': sum(asset_contributions.values())
        }
        
        # Benchmark comparison if provided
        if benchmark_weights is not None:
            benchmark_return = (returns * benchmark_weights).sum(axis=1).mean() * 252
            active_return = portfolio_return - benchmark_return
            
            # Asset allocation effect and security selection effect
            allocation_effect = {}
            selection_effect = {}
            
            for i, asset in enumerate(returns.columns):
                asset_return = returns[asset].mean() * 252
                
                # Allocation effect: (portfolio_weight - benchmark_weight) * benchmark_return
                weight_diff = weights[i] - benchmark_weights[i]
                allocation_effect[asset] = weight_diff * benchmark_return
                
                # Selection effect: benchmark_weight * (asset_return - benchmark_return)
                return_diff = asset_return - benchmark_return
                selection_effect[asset] = benchmark_weights[i] * return_diff
            
            result.update({
                'benchmark_return': benchmark_return,
                'active_return': active_return,
                'allocation_effect': allocation_effect,
                'selection_effect': selection_effect,
                'total_allocation_effect': sum(allocation_effect.values()),
                'total_selection_effect': sum(selection_effect.values())
            })
        
        return result
    
    def generate_performance_report(self, 
                                  returns: pd.Series,
                                  benchmark_returns: Optional[pd.Series] = None,
                                  portfolio_name: str = "Portfolio") -> str:
        """
        Generate a comprehensive performance report.
        
        Args:
            returns: Portfolio returns
            benchmark_returns: Benchmark returns (optional)
            portfolio_name: Name of the portfolio
        
        Returns:
            Formatted performance report string
        """
        metrics = self.calculate_metrics(returns, benchmark_returns)
        
        report = f"\n{portfolio_name} Performance Report\n"
        report += "=" * (len(portfolio_name) + 20) + "\n\n"
        
        # Return metrics
        report += "Return Metrics:\n"
        report += f"  Total Return: {metrics.get('total_return', 0):.2%}\n"
        report += f"  Annual Return: {metrics.get('annual_return', 0):.2%}\n"
        report += f"  Best Month: {metrics.get('best_month', 0):.2%}\n"
        report += f"  Worst Month: {metrics.get('worst_month', 0):.2%}\n\n"
        
        # Risk metrics
        report += "Risk Metrics:\n"
        report += f"  Annual Volatility: {metrics.get('annual_volatility', 0):.2%}\n"
        report += f"  Maximum Drawdown: {metrics.get('max_drawdown', 0):.2%}\n"
        report += f"  Skewness: {metrics.get('skewness', 0):.3f}\n"
        report += f"  Kurtosis: {metrics.get('kurtosis', 0):.3f}\n\n"
        
        # Risk-adjusted metrics
        report += "Risk-Adjusted Metrics:\n"
        report += f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}\n"
        report += f"  Sortino Ratio: {metrics.get('sortino_ratio', 0):.3f}\n"
        report += f"  Calmar Ratio: {metrics.get('calmar_ratio', 0):.3f}\n\n"
        
        # Benchmark comparison (if available)
        if benchmark_returns is not None:
            report += "Benchmark Comparison:\n"
            report += f"  Alpha: {metrics.get('alpha', 0):.2%}\n"
            report += f"  Beta: {metrics.get('beta', 0):.3f}\n"
            report += f"  R-Squared: {metrics.get('r_squared', 0):.3f}\n"
            report += f"  Tracking Error: {metrics.get('tracking_error', 0):.2%}\n"
            report += f"  Information Ratio: {metrics.get('information_ratio', 0):.3f}\n"
            report += f"  Up Capture Ratio: {metrics.get('up_capture_ratio', 0):.2%}\n"
            report += f"  Down Capture Ratio: {metrics.get('down_capture_ratio', 0):.2%}\n\n"
        
        return report
