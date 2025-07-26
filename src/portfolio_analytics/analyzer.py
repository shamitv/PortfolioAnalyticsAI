"""
Portfolio Analyzer for generating comprehensive metrics and visualizations for LLM analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from .portfolio import Portfolio
from .performance import PerformanceAnalyzer
from .risk_models import RiskModel
from .visualization import PortfolioVisualizer


class Analyzer:
    """
    Comprehensive portfolio analyzer for generating metrics, Greeks, and visualizations
    specifically designed to feed into Vision-Capable LLMs.
    
    This class takes a Portfolio object and generates:
    1. Comprehensive metrics and Greeks
    2. Visualization images optimized for LLM analysis
    """
    
    def __init__(self, portfolio: Portfolio):
        """
        Initialize the analyzer with a portfolio.
        
        Args:
            portfolio: Portfolio object to analyze
        """
        self.portfolio = portfolio
        self.performance_analyzer = PerformanceAnalyzer()
        self.risk_model = RiskModel()
        self.visualizer = PortfolioVisualizer()
        
        # Storage for generated content
        self.metrics = {}
        self.greeks = {}
        self.visualizations = {}
        
    def generate_comprehensive_analysis(self, 
                                      benchmark_returns: Optional[pd.Series] = None,
                                      risk_free_rate: float = 0.02) -> Dict[str, Any]:
        """
        Generate complete analysis including metrics, Greeks, and visualizations.
        
        Args:
            benchmark_returns: Benchmark returns for comparison
            risk_free_rate: Risk-free rate for calculations
            
        Returns:
            Dictionary containing all analysis results
        """
        if self.portfolio.returns is None:
            raise ValueError("Portfolio data not loaded. Call portfolio.load_data() first.")
        
        # Generate all metrics and Greeks
        self.generate_metrics(benchmark_returns, risk_free_rate)
        self.generate_greeks()
        
        # Generate all visualizations
        self.generate_visualizations(benchmark_returns)
        
        return {
            'metrics': self.metrics,
            'greeks': self.greeks,
            'visualizations': self.visualizations,
            'portfolio_summary': self._generate_portfolio_summary()
        }
    
    def generate_metrics(self, 
                        benchmark_returns: Optional[pd.Series] = None,
                        risk_free_rate: float = 0.02) -> Dict[str, Any]:
        """
        Generate comprehensive portfolio metrics.
        
        Args:
            benchmark_returns: Benchmark returns for comparison
            risk_free_rate: Risk-free rate
            
        Returns:
            Dictionary of metrics
        """
        portfolio_returns = (self.portfolio.returns * self.portfolio.weights).sum(axis=1)
        
        # Performance metrics
        performance_metrics = self.performance_analyzer.calculate_metrics(
            portfolio_returns, benchmark_returns, risk_free_rate
        )
        
        # Risk metrics
        risk_metrics = self.risk_model.calculate_risk_metrics(
            portfolio_returns, [0.95, 0.99], risk_free_rate
        )
        
        # Portfolio-specific metrics
        portfolio_metrics = self._calculate_portfolio_specific_metrics()
        
        # Asset allocation metrics
        allocation_metrics = self._calculate_allocation_metrics()
        
        # Time-based metrics
        time_metrics = self._calculate_time_based_metrics(portfolio_returns)
        
        self.metrics = {
            'performance': performance_metrics,
            'risk': risk_metrics,
            'portfolio': portfolio_metrics,
            'allocation': allocation_metrics,
            'time_based': time_metrics,
            'summary_statistics': self._calculate_summary_statistics(portfolio_returns)
        }
        
        return self.metrics
    
    def generate_greeks(self) -> Dict[str, Any]:
        """
        Generate portfolio Greeks and sensitivity metrics.
        
        Returns:
            Dictionary of Greeks and sensitivities
        """
        portfolio_returns = (self.portfolio.returns * self.portfolio.weights).sum(axis=1)
        
        # Delta (sensitivity to market movements) - approximated using beta
        market_proxy = self.portfolio.returns.mean(axis=1)  # Simple market proxy
        delta = np.cov(portfolio_returns, market_proxy)[0, 1] / np.var(market_proxy)
        
        # Gamma (convexity) - second derivative approximation
        gamma = self._calculate_gamma(portfolio_returns)
        
        # Theta (time decay) - using rolling performance
        theta = self._calculate_theta(portfolio_returns)
        
        # Vega (volatility sensitivity)
        vega = self._calculate_vega(portfolio_returns)
        
        # Rho (interest rate sensitivity)
        rho = self._calculate_rho(portfolio_returns)
        
        # Portfolio-specific Greeks
        portfolio_greeks = {
            'portfolio_delta': delta,
            'portfolio_gamma': gamma,
            'portfolio_theta': theta,
            'portfolio_vega': vega,
            'portfolio_rho': rho
        }
        
        # Asset-level Greeks
        asset_greeks = self._calculate_asset_greeks()
        
        self.greeks = {
            'portfolio_greeks': portfolio_greeks,
            'asset_greeks': asset_greeks,
            'sensitivity_analysis': self._calculate_sensitivity_analysis()
        }
        
        return self.greeks
    
    def generate_visualizations(self, benchmark_returns: Optional[pd.Series] = None) -> Dict[str, str]:
        """
        Generate all visualization images as base64 encoded strings for LLM consumption.
        
        Args:
            benchmark_returns: Benchmark returns for comparison
            
        Returns:
            Dictionary of base64 encoded images
        """
        portfolio_returns = (self.portfolio.returns * self.portfolio.weights).sum(axis=1)
        
        self.visualizations = {
            'price_history': self._create_price_history_chart(),
            'returns_distribution': self._create_returns_distribution_chart(portfolio_returns),
            'correlation_matrix': self._create_correlation_matrix_chart(),
            'portfolio_composition': self._create_portfolio_composition_chart(),
            'cumulative_returns': self._create_cumulative_returns_chart(portfolio_returns, benchmark_returns),
            'drawdown_analysis': self._create_drawdown_chart(portfolio_returns),
            'risk_return_scatter': self._create_risk_return_scatter(),
            'rolling_metrics': self._create_rolling_metrics_chart(portfolio_returns),
            'performance_heatmap': self._create_performance_heatmap(),
            'greek_sensitivity': self._create_greek_sensitivity_chart()
        }
        
        return self.visualizations
    
    def _calculate_portfolio_specific_metrics(self) -> Dict[str, float]:
        """Calculate portfolio-specific metrics."""
        return {
            'number_of_assets': len(self.portfolio.symbols),
            'portfolio_concentration': self._calculate_concentration(),
            'diversification_ratio': self._calculate_diversification_ratio(),
            'effective_number_of_assets': self._calculate_effective_assets(),
            'turnover_rate': self._calculate_turnover_rate(),
            'portfolio_beta': self._calculate_portfolio_beta()
        }
    
    def _calculate_allocation_metrics(self) -> Dict[str, Any]:
        """Calculate allocation-related metrics."""
        weights = self.portfolio.weights
        
        return {
            'weights': dict(zip(self.portfolio.symbols, weights)),
            'largest_position': np.max(weights),
            'smallest_position': np.min(weights),
            'weight_concentration_hhi': np.sum(weights ** 2),
            'number_positions_over_5pct': np.sum(weights > 0.05),
            'top_3_positions_weight': np.sum(np.sort(weights)[-3:])
        }
    
    def _calculate_time_based_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate time-based performance metrics."""
        if len(returns) == 0:
            return {}
        
        # Monthly analysis
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Quarterly analysis
        quarterly_returns = returns.resample('Q').apply(lambda x: (1 + x).prod() - 1)
        
        # Yearly analysis
        yearly_returns = returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
        
        return {
            'monthly_stats': {
                'mean': monthly_returns.mean(),
                'std': monthly_returns.std(),
                'skew': monthly_returns.skew(),
                'kurtosis': monthly_returns.kurtosis(),
                'best_month': monthly_returns.max(),
                'worst_month': monthly_returns.min(),
                'positive_months_pct': (monthly_returns > 0).mean()
            },
            'quarterly_stats': {
                'mean': quarterly_returns.mean(),
                'std': quarterly_returns.std(),
                'best_quarter': quarterly_returns.max(),
                'worst_quarter': quarterly_returns.min()
            },
            'yearly_stats': {
                'mean': yearly_returns.mean(),
                'std': yearly_returns.std(),
                'positive_years_pct': (yearly_returns > 0).mean() if len(yearly_returns) > 0 else 0
            }
        }
    
    def _calculate_summary_statistics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate summary statistics."""
        return {
            'count': len(returns),
            'mean_daily_return': returns.mean(),
            'median_daily_return': returns.median(),
            'std_daily_return': returns.std(),
            'min_daily_return': returns.min(),
            'max_daily_return': returns.max(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'jarque_bera_stat': self._jarque_bera_test(returns)
        }
    
    def _calculate_gamma(self, returns: pd.Series) -> float:
        """Calculate portfolio gamma (convexity measure)."""
        # Approximate gamma using second derivative of returns
        if len(returns) < 3:
            return 0.0
        
        # Calculate second differences
        first_diff = returns.diff()
        second_diff = first_diff.diff()
        
        return second_diff.std() / returns.std() if returns.std() != 0 else 0.0
    
    def _calculate_theta(self, returns: pd.Series) -> float:
        """Calculate portfolio theta (time decay)."""
        if len(returns) < 252:
            return 0.0
        
        # Rolling 252-day performance decay
        rolling_returns = returns.rolling(252).mean()
        theta = rolling_returns.diff().mean()
        
        return theta * 252  # Annualized
    
    def _calculate_vega(self, returns: pd.Series) -> float:
        """Calculate portfolio vega (volatility sensitivity)."""
        if len(returns) < 252:
            return 0.0
        
        # Sensitivity to volatility changes
        rolling_vol = returns.rolling(252).std()
        vol_change = rolling_vol.diff()
        return_change = returns.rolling(252).mean().diff()
        
        if vol_change.std() != 0:
            vega = np.cov(return_change.dropna(), vol_change.dropna())[0, 1] / vol_change.var()
        else:
            vega = 0.0
        
        return vega
    
    def _calculate_rho(self, returns: pd.Series) -> float:
        """Calculate portfolio rho (interest rate sensitivity)."""
        # Simplified rho calculation based on duration-like measure
        # Using inverse relationship with returns
        time_weights = np.arange(1, len(returns) + 1) / len(returns)
        weighted_return = np.sum(returns * time_weights)
        
        return -weighted_return * 100  # Scaled sensitivity
    
    def _calculate_asset_greeks(self) -> Dict[str, Dict[str, float]]:
        """Calculate Greeks for individual assets."""
        asset_greeks = {}
        
        for i, symbol in enumerate(self.portfolio.symbols):
            asset_returns = self.portfolio.returns[symbol]
            
            asset_greeks[symbol] = {
                'weight': self.portfolio.weights[i],
                'delta': asset_returns.corr(self.portfolio.returns.mean(axis=1)),
                'volatility': asset_returns.std() * np.sqrt(252),
                'contribution_to_risk': self.portfolio.weights[i] * asset_returns.std(),
                'beta': self._calculate_asset_beta(asset_returns),
                'alpha': self._calculate_asset_alpha(asset_returns)
            }
        
        return asset_greeks
    
    def _calculate_sensitivity_analysis(self) -> Dict[str, Any]:
        """Calculate portfolio sensitivity to various factors."""
        portfolio_returns = (self.portfolio.returns * self.portfolio.weights).sum(axis=1)
        
        # Sensitivity to market stress scenarios
        stress_scenarios = {
            'market_crash_10pct': self._stress_test_scenario(portfolio_returns, -0.10),
            'market_crash_20pct': self._stress_test_scenario(portfolio_returns, -0.20),
            'volatility_spike_50pct': self._volatility_stress_test(portfolio_returns, 1.5),
            'correlation_spike': self._correlation_stress_test()
        }
        
        return {
            'stress_scenarios': stress_scenarios,
            'factor_loadings': self._calculate_factor_loadings(),
            'regime_analysis': self._regime_analysis(portfolio_returns)
        }
    
    def _create_price_history_chart(self) -> str:
        """Create price history chart as base64 string."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for column in self.portfolio.data.columns:
            ax.plot(self.portfolio.data.index, self.portfolio.data[column], 
                   label=column, linewidth=2)
        
        ax.set_title('Portfolio Assets Price History', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return self._fig_to_base64(fig)
    
    def _create_returns_distribution_chart(self, returns: pd.Series) -> str:
        """Create returns distribution chart."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Histogram
        ax1.hist(returns, bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
        ax1.set_title('Returns Distribution')
        ax1.set_xlabel('Daily Returns')
        ax1.set_ylabel('Density')
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(returns, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot vs Normal Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Box plot
        ax3.boxplot(returns)
        ax3.set_title('Returns Box Plot')
        ax3.set_ylabel('Daily Returns')
        ax3.grid(True, alpha=0.3)
        
        # Rolling volatility
        rolling_vol = returns.rolling(30).std() * np.sqrt(252)
        ax4.plot(rolling_vol.index, rolling_vol)
        ax4.set_title('Rolling 30-Day Volatility (Annualized)')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Volatility')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_correlation_matrix_chart(self) -> str:
        """Create correlation matrix heatmap."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        correlation_matrix = self.portfolio.returns.corr()
        
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   cmap='RdYlBu_r', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={'label': 'Correlation'},
                   ax=ax)
        
        ax.set_title('Asset Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_portfolio_composition_chart(self) -> str:
        """Create portfolio composition chart."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pie chart
        ax1.pie(self.portfolio.weights, labels=self.portfolio.symbols, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Portfolio Composition (Pie Chart)')
        
        # Bar chart
        bars = ax2.bar(self.portfolio.symbols, self.portfolio.weights)
        ax2.set_title('Portfolio Composition (Bar Chart)')
        ax2.set_ylabel('Weight')
        ax2.set_xlabel('Assets')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1%}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_cumulative_returns_chart(self, returns: pd.Series, benchmark_returns: Optional[pd.Series]) -> str:
        """Create cumulative returns chart."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        cumulative_returns = (1 + returns).cumprod()
        ax.plot(cumulative_returns.index, cumulative_returns, 
               label='Portfolio', linewidth=2, color='blue')
        
        if benchmark_returns is not None:
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            ax.plot(benchmark_cumulative.index, benchmark_cumulative, 
                   label='Benchmark', linewidth=2, linestyle='--', color='red')
        
        ax.set_title('Cumulative Returns Comparison', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_drawdown_chart(self, returns: pd.Series) -> str:
        """Create drawdown analysis chart."""
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Cumulative returns with peaks
        ax1.plot(cumulative_returns.index, cumulative_returns, 
                label='Cumulative Returns', color='blue')
        ax1.plot(running_max.index, running_max, 
                label='Peak', color='red', linestyle='--', alpha=0.7)
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Portfolio Drawdown Analysis', fontweight='bold')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        # Drawdown
        ax2.fill_between(drawdown.index, drawdown, 0, 
                        color='red', alpha=0.3, label='Drawdown')
        ax2.plot(drawdown.index, drawdown, color='red')
        ax2.set_ylabel('Drawdown')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_risk_return_scatter(self) -> str:
        """Create risk-return scatter plot for assets."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        asset_returns = self.portfolio.returns.mean() * 252
        asset_volatilities = self.portfolio.returns.std() * np.sqrt(252)
        
        scatter = ax.scatter(asset_volatilities, asset_returns, 
                           s=self.portfolio.weights * 1000, 
                           alpha=0.6, c=range(len(self.portfolio.symbols)), 
                           cmap='viridis')
        
        for i, symbol in enumerate(self.portfolio.symbols):
            ax.annotate(symbol, (asset_volatilities[i], asset_returns[i]))
        
        ax.set_title('Risk-Return Profile by Asset', fontsize=16, fontweight='bold')
        ax.set_xlabel('Volatility (Risk)')
        ax.set_ylabel('Expected Return')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_rolling_metrics_chart(self, returns: pd.Series) -> str:
        """Create rolling metrics chart."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Rolling returns
        rolling_returns = returns.rolling(252).mean() * 252
        ax1.plot(rolling_returns.index, rolling_returns)
        ax1.set_title('Rolling 1-Year Returns')
        ax1.set_ylabel('Annual Return')
        ax1.grid(True, alpha=0.3)
        
        # Rolling volatility
        rolling_vol = returns.rolling(252).std() * np.sqrt(252)
        ax2.plot(rolling_vol.index, rolling_vol)
        ax2.set_title('Rolling 1-Year Volatility')
        ax2.set_ylabel('Annual Volatility')
        ax2.grid(True, alpha=0.3)
        
        # Rolling Sharpe ratio
        rolling_sharpe = (rolling_returns - 0.02) / rolling_vol
        ax3.plot(rolling_sharpe.index, rolling_sharpe)
        ax3.set_title('Rolling 1-Year Sharpe Ratio')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.grid(True, alpha=0.3)
        
        # Rolling maximum drawdown
        def rolling_max_drawdown(series):
            cumulative = (1 + series).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return abs(drawdown.min())
        
        rolling_mdd = returns.rolling(252).apply(rolling_max_drawdown)
        ax4.plot(rolling_mdd.index, rolling_mdd)
        ax4.set_title('Rolling 1-Year Maximum Drawdown')
        ax4.set_ylabel('Maximum Drawdown')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_performance_heatmap(self) -> str:
        """Create monthly performance heatmap."""
        portfolio_returns = (self.portfolio.returns * self.portfolio.weights).sum(axis=1)
        
        # Create monthly returns matrix
        monthly_returns = portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        if len(monthly_returns) == 0:
            # Create empty chart if no data
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, 'No monthly data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Monthly Performance Heatmap')
            return self._fig_to_base64(fig)
        
        # Create year-month matrix
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        monthly_data = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).first()
        
        years = sorted(monthly_returns.index.year.unique())
        months = range(1, 13)
        
        heatmap_data = pd.DataFrame(index=years, columns=months)
        
        for year in years:
            for month in months:
                try:
                    heatmap_data.loc[year, month] = monthly_data.loc[(year, month)]
                except KeyError:
                    heatmap_data.loc[year, month] = np.nan
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(heatmap_data.astype(float), 
                   annot=True, 
                   fmt='.1%',
                   cmap='RdYlGn', 
                   center=0,
                   cbar_kws={'label': 'Monthly Return'},
                   ax=ax)
        
        ax.set_title('Monthly Performance Heatmap', fontsize=16, fontweight='bold')
        ax.set_xlabel('Month')
        ax.set_ylabel('Year')
        
        # Set month labels
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticklabels(month_labels)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_greek_sensitivity_chart(self) -> str:
        """Create Greeks sensitivity analysis chart."""
        if not self.greeks:
            self.generate_greeks()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Portfolio Greeks
        greek_names = list(self.greeks['portfolio_greeks'].keys())
        greek_values = list(self.greeks['portfolio_greeks'].values())
        
        ax1.bar(greek_names, greek_values)
        ax1.set_title('Portfolio Greeks')
        ax1.set_ylabel('Value')
        plt.setp(ax1.get_xticklabels(), rotation=45)
        
        # Asset contributions to portfolio risk
        asset_greeks = self.greeks['asset_greeks']
        symbols = list(asset_greeks.keys())
        risk_contributions = [asset_greeks[symbol]['contribution_to_risk'] for symbol in symbols]
        
        ax2.bar(symbols, risk_contributions)
        ax2.set_title('Asset Risk Contributions')
        ax2.set_ylabel('Risk Contribution')
        plt.setp(ax2.get_xticklabels(), rotation=45)
        
        # Asset volatilities
        volatilities = [asset_greeks[symbol]['volatility'] for symbol in symbols]
        ax3.bar(symbols, volatilities)
        ax3.set_title('Asset Volatilities')
        ax3.set_ylabel('Annual Volatility')
        plt.setp(ax3.get_xticklabels(), rotation=45)
        
        # Asset betas
        betas = [asset_greeks[symbol]['beta'] for symbol in symbols]
        ax4.bar(symbols, betas)
        ax4.set_title('Asset Betas')
        ax4.set_ylabel('Beta')
        plt.setp(ax4.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        return image_base64
    
    def _generate_portfolio_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive portfolio summary."""
        return {
            'portfolio_name': self.portfolio.name,
            'number_of_assets': len(self.portfolio.symbols),
            'assets': self.portfolio.symbols,
            'weights': dict(zip(self.portfolio.symbols, self.portfolio.weights)),
            'data_start_date': str(self.portfolio.data.index.min().date()) if self.portfolio.data is not None else None,
            'data_end_date': str(self.portfolio.data.index.max().date()) if self.portfolio.data is not None else None,
            'total_observations': len(self.portfolio.returns) if self.portfolio.returns is not None else 0,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    # Helper methods for metrics calculations
    def _calculate_concentration(self) -> float:
        """Calculate portfolio concentration using Herfindahl-Hirschman Index."""
        return np.sum(self.portfolio.weights ** 2)
    
    def _calculate_diversification_ratio(self) -> float:
        """Calculate diversification ratio."""
        weighted_vol = np.sum(self.portfolio.weights * self.portfolio.returns.std() * np.sqrt(252))
        portfolio_vol = ((self.portfolio.returns * self.portfolio.weights).sum(axis=1)).std() * np.sqrt(252)
        return weighted_vol / portfolio_vol if portfolio_vol != 0 else 0
    
    def _calculate_effective_assets(self) -> float:
        """Calculate effective number of assets."""
        return 1 / np.sum(self.portfolio.weights ** 2)
    
    def _calculate_turnover_rate(self) -> float:
        """Calculate portfolio turnover rate (simplified)."""
        # Simplified calculation - in practice would need rebalancing data
        return 0.0  # Placeholder
    
    def _calculate_portfolio_beta(self) -> float:
        """Calculate portfolio beta against equal-weighted market."""
        portfolio_returns = (self.portfolio.returns * self.portfolio.weights).sum(axis=1)
        market_returns = self.portfolio.returns.mean(axis=1)
        
        covariance = np.cov(portfolio_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        
        return covariance / market_variance if market_variance != 0 else 0
    
    def _calculate_asset_beta(self, asset_returns: pd.Series) -> float:
        """Calculate individual asset beta."""
        market_returns = self.portfolio.returns.mean(axis=1)
        covariance = np.cov(asset_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        return covariance / market_variance if market_variance != 0 else 0
    
    def _calculate_asset_alpha(self, asset_returns: pd.Series) -> float:
        """Calculate individual asset alpha."""
        market_returns = self.portfolio.returns.mean(axis=1)
        beta = self._calculate_asset_beta(asset_returns)
        alpha = (asset_returns.mean() - beta * market_returns.mean()) * 252
        return alpha
    
    def _jarque_bera_test(self, returns: pd.Series) -> float:
        """Calculate Jarque-Bera test statistic."""
        n = len(returns)
        if n < 3:
            return 0.0
        
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        jb_stat = (n / 6) * (skewness**2 + (1/4) * (kurtosis**2))
        return jb_stat
    
    def _stress_test_scenario(self, returns: pd.Series, shock: float) -> Dict[str, float]:
        """Perform stress test scenario."""
        shocked_returns = returns + shock
        return {
            'scenario_return': shocked_returns.mean() * 252,
            'scenario_volatility': shocked_returns.std() * np.sqrt(252),
            'var_95': np.percentile(shocked_returns, 5),
            'expected_shortfall': shocked_returns[shocked_returns <= np.percentile(shocked_returns, 5)].mean()
        }
    
    def _volatility_stress_test(self, returns: pd.Series, vol_multiplier: float) -> Dict[str, float]:
        """Perform volatility stress test."""
        base_vol = returns.std()
        stressed_vol = base_vol * vol_multiplier
        
        return {
            'base_volatility': base_vol * np.sqrt(252),
            'stressed_volatility': stressed_vol * np.sqrt(252),
            'volatility_impact': (stressed_vol - base_vol) * np.sqrt(252)
        }
    
    def _correlation_stress_test(self) -> Dict[str, float]:
        """Perform correlation stress test."""
        correlation_matrix = self.portfolio.returns.corr()
        
        # Create stressed correlation matrix (increase all correlations)
        stressed_corr = correlation_matrix * 1.5
        np.fill_diagonal(stressed_corr.values, 1.0)
        
        return {
            'average_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean(),
            'max_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].max(),
            'min_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].min()
        }
    
    def _calculate_factor_loadings(self) -> Dict[str, float]:
        """Calculate factor loadings (simplified)."""
        portfolio_returns = (self.portfolio.returns * self.portfolio.weights).sum(axis=1)
        
        # Market factor (simplified)
        market_factor = self.portfolio.returns.mean(axis=1)
        market_loading = np.corrcoef(portfolio_returns, market_factor)[0, 1]
        
        return {
            'market_loading': market_loading,
            'r_squared': market_loading ** 2
        }
    
    def _regime_analysis(self, returns: pd.Series) -> Dict[str, Any]:
        """Perform regime analysis."""
        # Simple regime identification based on volatility
        rolling_vol = returns.rolling(30).std()
        vol_threshold = rolling_vol.quantile(0.7)
        
        high_vol_periods = rolling_vol > vol_threshold
        low_vol_periods = rolling_vol <= vol_threshold
        
        return {
            'high_volatility_regime': {
                'periods': high_vol_periods.sum(),
                'avg_return': returns[high_vol_periods].mean() * 252,
                'avg_volatility': returns[high_vol_periods].std() * np.sqrt(252)
            },
            'low_volatility_regime': {
                'periods': low_vol_periods.sum(),
                'avg_return': returns[low_vol_periods].mean() * 252,
                'avg_volatility': returns[low_vol_periods].std() * np.sqrt(252)
            }
        }
    
    def export_for_llm(self, 
                      benchmark_returns: Optional[pd.Series] = None,
                      risk_free_rate: float = 0.02,
                      output_format: str = "comprehensive") -> Dict[str, Any]:
        """
        Export all analysis results in a format optimized for LLM consumption.
        
        Args:
            benchmark_returns: Benchmark returns for comparison
            risk_free_rate: Risk-free rate
            output_format: Export format ('comprehensive', 'summary', 'metrics_only', 'visuals_only')
            
        Returns:
            Dictionary formatted for LLM analysis
        """
        if output_format == "comprehensive":
            return self.generate_comprehensive_analysis(benchmark_returns, risk_free_rate)
        elif output_format == "summary":
            return {
                'portfolio_summary': self._generate_portfolio_summary(),
                'key_metrics': self._extract_key_metrics(),
                'visualizations': self.generate_visualizations(benchmark_returns)
            }
        elif output_format == "metrics_only":
            return self.generate_metrics(benchmark_returns, risk_free_rate)
        elif output_format == "visuals_only":
            return self.generate_visualizations(benchmark_returns)
        else:
            raise ValueError(f"Unknown output format: {output_format}")
    
    def _extract_key_metrics(self) -> Dict[str, float]:
        """Extract key metrics for summary view."""
        if not self.metrics:
            self.generate_metrics()
        
        return {
            'annual_return': self.metrics['performance'].get('annual_return', 0),
            'annual_volatility': self.metrics['performance'].get('annual_volatility', 0),
            'sharpe_ratio': self.metrics['performance'].get('sharpe_ratio', 0),
            'max_drawdown': self.metrics['performance'].get('max_drawdown', 0),
            'var_95': self.metrics['risk'].get('var_95', 0),
            'beta': self.metrics['portfolio'].get('portfolio_beta', 0),
            'number_of_assets': self.metrics['portfolio'].get('number_of_assets', 0),
            'concentration': self.metrics['portfolio'].get('portfolio_concentration', 0)
        }
    
    def generate_openai_messages(self, 
                               benchmark_returns: Optional[pd.Series] = None,
                               risk_free_rate: float = 0.02,
                               analysis_request: str = "Analyze this portfolio and provide investment insights",
                               include_visualizations: bool = True,
                               output_format: str = "comprehensive") -> List[Dict[str, Any]]:
        """
        Generate OpenAI-compatible messages for LLM analysis.
        
        Args:
            benchmark_returns: Benchmark returns for comparison
            risk_free_rate: Risk-free rate for calculations
            analysis_request: The analysis request/prompt for the LLM
            include_visualizations: Whether to include base64 images for vision models
            output_format: Export format ('comprehensive', 'summary', 'metrics_only', 'visuals_only')
            
        Returns:
            List of message dictionaries compatible with OpenAI API
        """
        # Generate the complete analysis
        analysis_data = self.export_for_llm(benchmark_returns, risk_free_rate, output_format)
        
        messages = []
        
        # System message with context about the analysis
        system_message = {
            "role": "system",
            "content": self._generate_system_prompt()
        }
        messages.append(system_message)
        
        # User message with the analysis request and data
        user_content = []
        
        # Add the text analysis request
        user_content.append({
            "type": "text",
            "text": f"{analysis_request}\n\n{self._format_analysis_for_llm(analysis_data, include_visualizations)}"
        })
        
        # Add visualizations as images if requested and available
        if include_visualizations and 'visualizations' in analysis_data:
            for viz_name, base64_image in analysis_data['visualizations'].items():
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        "detail": "high"
                    }
                })
        
        user_message = {
            "role": "user",
            "content": user_content
        }
        messages.append(user_message)
        
        return messages
    
    def _generate_system_prompt(self) -> str:
        """Generate system prompt for the LLM."""
        return """You are an expert portfolio analyst and investment advisor with deep knowledge of:

- Modern Portfolio Theory and asset allocation
- Risk management and Value at Risk (VaR) analysis  
- Portfolio optimization techniques
- Financial metrics and performance analysis
- Technical and fundamental analysis
- Market dynamics and economic factors

You will receive comprehensive portfolio analysis data including:
- Performance metrics (returns, Sharpe ratio, volatility, etc.)
- Risk metrics (VaR, Expected Shortfall, maximum drawdown, etc.)
- Portfolio Greeks (Delta, Gamma, Theta, Vega, Rho)
- Asset allocation and concentration analysis
- Time-based performance analysis
- Stress testing and scenario analysis
- Multiple visualization charts

Your task is to:
1. Analyze the quantitative data thoroughly
2. Interpret the visual charts and patterns
3. Identify strengths and weaknesses in the portfolio
4. Provide actionable investment recommendations
5. Suggest optimization opportunities
6. Highlight potential risks and mitigation strategies

Provide clear, professional analysis with specific recommendations backed by the data."""
    
    def _format_analysis_for_llm(self, analysis_data: Dict[str, Any], include_visualizations: bool = True) -> str:
        """Format analysis data into text for LLM consumption."""
        
        formatted_text = []
        
        # Portfolio Summary
        if 'portfolio_summary' in analysis_data:
            summary = analysis_data['portfolio_summary']
            formatted_text.append("=== PORTFOLIO SUMMARY ===")
            formatted_text.append(f"Portfolio Name: {summary.get('portfolio_name', 'N/A')}")
            formatted_text.append(f"Assets: {', '.join(summary.get('assets', []))}")
            formatted_text.append(f"Analysis Period: {summary.get('data_start_date', 'N/A')} to {summary.get('data_end_date', 'N/A')}")
            formatted_text.append(f"Total Observations: {summary.get('total_observations', 0):,} trading days")
            formatted_text.append("")
        
        # Key Metrics
        if 'metrics' in analysis_data:
            formatted_text.append("=== PERFORMANCE METRICS ===")
            
            # Performance metrics
            if 'performance' in analysis_data['metrics']:
                perf = analysis_data['metrics']['performance']
                formatted_text.append(f"Total Return: {perf.get('total_return', 0):.2%}")
                formatted_text.append(f"Annual Return: {perf.get('annual_return', 0):.2%}")
                formatted_text.append(f"Annual Volatility: {perf.get('annual_volatility', 0):.2%}")
                formatted_text.append(f"Sharpe Ratio: {perf.get('sharpe_ratio', 0):.3f}")
                formatted_text.append(f"Sortino Ratio: {perf.get('sortino_ratio', 0):.3f}")
                formatted_text.append(f"Maximum Drawdown: {perf.get('max_drawdown', 0):.2%}")
                formatted_text.append("")
            
            # Risk metrics
            if 'risk' in analysis_data['metrics']:
                risk = analysis_data['metrics']['risk']
                formatted_text.append("=== RISK METRICS ===")
                formatted_text.append(f"VaR (95%): {risk.get('var_95', 0):.2%}")
                formatted_text.append(f"Expected Shortfall (95%): {risk.get('expected_shortfall_95', 0):.2%}")
                formatted_text.append(f"Skewness: {risk.get('skewness', 0):.3f}")
                formatted_text.append(f"Kurtosis: {risk.get('kurtosis', 0):.3f}")
                formatted_text.append("")
            
            # Portfolio composition
            if 'allocation' in analysis_data['metrics']:
                alloc = analysis_data['metrics']['allocation']
                formatted_text.append("=== PORTFOLIO COMPOSITION ===")
                if 'weights' in alloc:
                    for asset, weight in alloc['weights'].items():
                        formatted_text.append(f"{asset}: {weight:.1%}")
                formatted_text.append(f"Concentration (HHI): {alloc.get('weight_concentration_hhi', 0):.3f}")
                formatted_text.append(f"Largest Position: {alloc.get('largest_position', 0):.1%}")
                formatted_text.append("")
        
        # Portfolio Greeks
        if 'greeks' in analysis_data and 'portfolio_greeks' in analysis_data['greeks']:
            greeks = analysis_data['greeks']['portfolio_greeks']
            formatted_text.append("=== PORTFOLIO GREEKS ===")
            formatted_text.append(f"Delta (Market Sensitivity): {greeks.get('portfolio_delta', 0):.3f}")
            formatted_text.append(f"Gamma (Convexity): {greeks.get('portfolio_gamma', 0):.3f}")
            formatted_text.append(f"Theta (Time Decay): {greeks.get('portfolio_theta', 0):.3f}")
            formatted_text.append(f"Vega (Volatility Sensitivity): {greeks.get('portfolio_vega', 0):.3f}")
            formatted_text.append(f"Rho (Interest Rate Sensitivity): {greeks.get('portfolio_rho', 0):.3f}")
            formatted_text.append("")
        
        # Asset-level analysis
        if 'greeks' in analysis_data and 'asset_greeks' in analysis_data['greeks']:
            asset_greeks = analysis_data['greeks']['asset_greeks']
            formatted_text.append("=== ASSET-LEVEL ANALYSIS ===")
            formatted_text.append("Asset | Weight | Beta | Alpha | Volatility | Risk Contribution")
            formatted_text.append("-" * 65)
            for asset, data in asset_greeks.items():
                formatted_text.append(
                    f"{asset:5} | {data.get('weight', 0):5.1%} | "
                    f"{data.get('beta', 0):4.2f} | {data.get('alpha', 0):5.2f} | "
                    f"{data.get('volatility', 0):9.1%} | {data.get('contribution_to_risk', 0):15.4f}"
                )
            formatted_text.append("")
        
        # Stress testing results
        if 'greeks' in analysis_data and 'sensitivity_analysis' in analysis_data['greeks']:
            sensitivity = analysis_data['greeks']['sensitivity_analysis']
            if 'stress_scenarios' in sensitivity:
                scenarios = sensitivity['stress_scenarios']
                formatted_text.append("=== STRESS TEST SCENARIOS ===")
                
                if 'market_crash_10pct' in scenarios:
                    crash_10 = scenarios['market_crash_10pct']
                    formatted_text.append(f"Market Crash -10%:")
                    formatted_text.append(f"  Scenario Return: {crash_10.get('scenario_return', 0):.2%}")
                    formatted_text.append(f"  Scenario Volatility: {crash_10.get('scenario_volatility', 0):.2%}")
                
                if 'volatility_spike_50pct' in scenarios:
                    vol_spike = scenarios['volatility_spike_50pct']
                    formatted_text.append(f"Volatility Spike +50%:")
                    formatted_text.append(f"  Base Volatility: {vol_spike.get('base_volatility', 0):.2%}")
                    formatted_text.append(f"  Stressed Volatility: {vol_spike.get('stressed_volatility', 0):.2%}")
                formatted_text.append("")
        
        # Visualization summary
        if include_visualizations and 'visualizations' in analysis_data:
            viz_count = len(analysis_data['visualizations'])
            formatted_text.append(f"=== VISUALIZATIONS INCLUDED ===")
            formatted_text.append(f"Total Charts: {viz_count}")
            formatted_text.append("Available Charts:")
            for viz_name in analysis_data['visualizations'].keys():
                formatted_text.append(f"  • {viz_name.replace('_', ' ').title()}")
            formatted_text.append("")
            formatted_text.append("Note: All charts are provided as high-resolution images for visual analysis.")
        
        return "\n".join(formatted_text)
    
    def generate_simple_message(self, 
                              analysis_request: str = "Analyze this portfolio and provide investment insights",
                              benchmark_returns: Optional[pd.Series] = None,
                              risk_free_rate: float = 0.02) -> List[Dict[str, str]]:
        """
        Generate simple text-only messages for non-vision LLM models.
        
        Args:
            analysis_request: The analysis request/prompt for the LLM
            benchmark_returns: Benchmark returns for comparison
            risk_free_rate: Risk-free rate for calculations
            
        Returns:
            List of simple message dictionaries (text only)
        """
        # Generate analysis without visualizations
        analysis_data = self.export_for_llm(benchmark_returns, risk_free_rate, "metrics_only")
        
        messages = [
            {
                "role": "system", 
                "content": self._generate_system_prompt()
            },
            {
                "role": "user",
                "content": f"{analysis_request}\n\n{self._format_analysis_for_llm(analysis_data, include_visualizations=False)}"
            }
        ]
        
        return messages
