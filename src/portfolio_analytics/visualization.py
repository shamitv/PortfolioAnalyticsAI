"""
Visualization tools for portfolio analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Tuple, Union
from .data_provider import DataProvider


class PortfolioVisualizer:
    """
    Comprehensive visualization tools for portfolio analysis.
    
    Provides static and interactive charts for portfolio performance,
    risk analysis, and optimization results with risk-free rate integration.
    """
    
    def __init__(self, style: str = "seaborn", data_provider: Optional[DataProvider] = None):
        """
        Initialize the visualizer.
        
        Args:
            style: Matplotlib style ('seaborn', 'ggplot', 'bmh', etc.)
            data_provider: DataProvider instance for fetching risk-free rates
        """
        plt.style.use(style if style in plt.style.available else 'default')
        self.color_palette = sns.color_palette("husl", 10)
        self.data_provider = data_provider
    
    def _get_risk_free_rate(self, 
                           start_date: str, 
                           end_date: str, 
                           annualized: bool = True) -> Optional[Union[float, pd.Series]]:
        """
        Get risk-free rate data for the specified period.
        
        Args:
            start_date: Start date for risk-free rate data
            end_date: End date for risk-free rate data
            annualized: Whether to return annualized rate
            
        Returns:
            Risk-free rate as float (mean) or Series (time series)
        """
        if not self.data_provider:
            return None
            
        try:
            risk_free_metadata = self.data_provider.get_risk_free_rate_metadata()
            risk_free_symbol = risk_free_metadata['symbol']
            
            rf_data = self.data_provider.get_price_data(
                symbols=risk_free_symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            if rf_data.empty:
                return None
                
            # Convert yield percentage to decimal
            rf_series = rf_data[risk_free_symbol] / 100
            
            if annualized:
                return rf_series
            else:
                # Convert to daily rate
                return rf_series / 252
                
        except Exception as e:
            print(f"Could not fetch risk-free rate: {e}")
            return None
    
    def plot_price_history(self, 
                          price_data: pd.DataFrame,
                          title: str = "Price History",
                          figsize: Tuple[int, int] = (12, 6),
                          interactive: bool = False) -> None:
        """
        Plot historical price data.
        
        Args:
            price_data: DataFrame with price data
            title: Chart title
            figsize: Figure size for matplotlib
            interactive: Whether to create interactive plotly chart
        """
        if interactive:
            self._plot_price_history_plotly(price_data, title)
        else:
            self._plot_price_history_matplotlib(price_data, title, figsize)
    
    def _plot_price_history_matplotlib(self, 
                                     price_data: pd.DataFrame,
                                     title: str,
                                     figsize: Tuple[int, int]) -> None:
        """Create matplotlib price history chart."""
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, column in enumerate(price_data.columns):
            ax.plot(price_data.index, price_data[column], 
                   label=column, color=self.color_palette[i % len(self.color_palette)])
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_price_history_plotly(self, 
                                 price_data: pd.DataFrame,
                                 title: str) -> None:
        """Create interactive plotly price history chart."""
        fig = go.Figure()
        
        for column in price_data.columns:
            fig.add_trace(go.Scatter(
                x=price_data.index,
                y=price_data[column],
                mode='lines',
                name=column,
                hovertemplate=f'{column}: $%{{y:.2f}}<br>Date: %{{x}}<extra></extra>'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified',
            template='plotly_white'
        )
        
        fig.show()
    
    def plot_returns_distribution(self, 
                                returns: pd.Series,
                                title: str = "Returns Distribution",
                                bins: int = 50,
                                figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot distribution of returns with normal overlay.
        
        Args:
            returns: Series of returns
            title: Chart title
            bins: Number of histogram bins
            figsize: Figure size
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram with normal overlay
        ax1.hist(returns, bins=bins, alpha=0.7, density=True, color='skyblue', edgecolor='black')
        
        # Normal distribution overlay
        mu, sigma = returns.mean(), returns.std()
        x = np.linspace(returns.min(), returns.max(), 100)
        normal_dist = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        ax1.plot(x, normal_dist, 'r-', linewidth=2, label='Normal Distribution')
        
        ax1.set_title(f'{title} - Histogram', fontweight='bold')
        ax1.set_xlabel('Returns')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(returns, dist="norm", plot=ax2)
        ax2.set_title(f'{title} - Q-Q Plot', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_matrix(self, 
                              returns: pd.DataFrame,
                              title: str = "Correlation Matrix",
                              figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Plot correlation matrix heatmap.
        
        Args:
            returns: DataFrame of asset returns
            title: Chart title
            figsize: Figure size
        """
        correlation_matrix = returns.corr()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   cmap='RdYlBu_r', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={'label': 'Correlation'},
                   ax=ax)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_efficient_frontier(self, 
                              returns_array: np.ndarray,
                              volatilities_array: np.ndarray,
                              sharpe_ratios_array: np.ndarray,
                              optimal_portfolio: Optional[Dict] = None,
                              title: str = "Efficient Frontier",
                              figsize: Tuple[int, int] = (12, 8),
                              interactive: bool = False,
                              risk_free_rate: Optional[float] = None,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None) -> None:
        """
        Plot efficient frontier with optional risk-free rate integration.
        
        Args:
            returns_array: Array of portfolio returns
            volatilities_array: Array of portfolio volatilities
            sharpe_ratios_array: Array of Sharpe ratios
            optimal_portfolio: Dictionary with optimal portfolio metrics
            title: Chart title
            figsize: Figure size
            interactive: Whether to create interactive chart
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            start_date: Start date for fetching risk-free rate data
            end_date: End date for fetching risk-free rate data
        """
        # Get risk-free rate if not provided but dates are available
        if risk_free_rate is None and start_date and end_date:
            rf_data = self._get_risk_free_rate(start_date, end_date, annualized=True)
            if rf_data is not None:
                risk_free_rate = rf_data.mean() if hasattr(rf_data, 'mean') else rf_data
        
        # Recalculate Sharpe ratios with actual risk-free rate if available
        if risk_free_rate is not None:
            excess_returns = returns_array - risk_free_rate
            updated_sharpe_ratios = excess_returns / volatilities_array
            title += f" (Risk-Free Rate: {risk_free_rate:.2%})"
        else:
            updated_sharpe_ratios = sharpe_ratios_array
        
        if interactive:
            self._plot_efficient_frontier_plotly(
                returns_array, volatilities_array, updated_sharpe_ratios, 
                optimal_portfolio, title, risk_free_rate
            )
        else:
            self._plot_efficient_frontier_matplotlib(
                returns_array, volatilities_array, updated_sharpe_ratios,
                optimal_portfolio, title, figsize, risk_free_rate
            )
    
    def _plot_efficient_frontier_matplotlib(self, 
                                          returns_array: np.ndarray,
                                          volatilities_array: np.ndarray,
                                          sharpe_ratios_array: np.ndarray,
                                          optimal_portfolio: Optional[Dict],
                                          title: str,
                                          figsize: Tuple[int, int],
                                          risk_free_rate: Optional[float] = None) -> None:
        """Create matplotlib efficient frontier chart."""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot efficient frontier
        scatter = ax.scatter(volatilities_array, returns_array, 
                           c=sharpe_ratios_array, cmap='viridis', 
                           alpha=0.6, s=50)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Sharpe Ratio', rotation=270, labelpad=20)
        
        # Add Capital Allocation Line (CAL) if risk-free rate is available
        if risk_free_rate is not None and optimal_portfolio:
            # Plot line from risk-free rate to optimal portfolio
            max_vol = max(volatilities_array) * 1.2
            cal_x = np.linspace(0, max_vol, 100)
            optimal_vol = optimal_portfolio['annual_volatility']
            optimal_ret = optimal_portfolio['expected_annual_return']
            slope = (optimal_ret - risk_free_rate) / optimal_vol
            cal_y = risk_free_rate + slope * cal_x
            
            ax.plot(cal_x, cal_y, 'r--', linewidth=2, alpha=0.8, 
                   label=f'Capital Allocation Line (RF: {risk_free_rate:.2%})')
            ax.scatter(0, risk_free_rate, marker='o', color='green', s=100, 
                      label=f'Risk-Free Rate ({risk_free_rate:.2%})', zorder=5)
        
        # Highlight optimal portfolio if provided
        if optimal_portfolio:
            ax.scatter(optimal_portfolio['annual_volatility'], 
                      optimal_portfolio['expected_annual_return'],
                      marker='*', color='red', s=500, 
                      label=f"Max Sharpe Ratio: {optimal_portfolio['sharpe_ratio']:.3f}")
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Volatility (Risk)', fontsize=12)
        ax.set_ylabel('Expected Return', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_efficient_frontier_plotly(self, 
                                      returns_array: np.ndarray,
                                      volatilities_array: np.ndarray,
                                      sharpe_ratios_array: np.ndarray,
                                      optimal_portfolio: Optional[Dict],
                                      title: str,
                                      risk_free_rate: Optional[float] = None) -> None:
        """Create interactive plotly efficient frontier chart."""
        fig = go.Figure()
        
        # Efficient frontier
        fig.add_trace(go.Scatter(
            x=volatilities_array,
            y=returns_array,
            mode='markers',
            marker=dict(
                size=8,
                color=sharpe_ratios_array,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Sharpe Ratio")
            ),
            text=[f"Return: {r:.2%}<br>Risk: {v:.2%}<br>Sharpe: {s:.3f}" 
                  for r, v, s in zip(returns_array, volatilities_array, sharpe_ratios_array)],
            hovertemplate='%{text}<extra></extra>',
            name='Efficient Frontier'
        ))
        
        # Add Capital Allocation Line if risk-free rate is available
        if risk_free_rate is not None and optimal_portfolio:
            max_vol = max(volatilities_array) * 1.2
            cal_x = np.linspace(0, max_vol, 100)
            optimal_vol = optimal_portfolio['annual_volatility']
            optimal_ret = optimal_portfolio['expected_annual_return']
            slope = (optimal_ret - risk_free_rate) / optimal_vol
            cal_y = risk_free_rate + slope * cal_x
            
            fig.add_trace(go.Scatter(
                x=cal_x,
                y=cal_y,
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                name=f'Capital Allocation Line',
                hovertemplate=f'CAL: Risk-Free Rate = {risk_free_rate:.2%}<br>Risk: %{{x:.2%}}<br>Expected Return: %{{y:.2%}}<extra></extra>'
            ))
            
            # Risk-free rate point
            fig.add_trace(go.Scatter(
                x=[0],
                y=[risk_free_rate],
                mode='markers',
                marker=dict(size=12, color='green', symbol='circle'),
                name=f'Risk-Free Rate ({risk_free_rate:.2%})',
                hovertemplate=f'Risk-Free Asset<br>Risk: 0%<br>Return: {risk_free_rate:.2%}<extra></extra>'
            ))
        
        # Optimal portfolio
        if optimal_portfolio:
            fig.add_trace(go.Scatter(
                x=[optimal_portfolio['annual_volatility']],
                y=[optimal_portfolio['expected_annual_return']],
                mode='markers',
                marker=dict(size=20, color='red', symbol='star'),
                name=f"Max Sharpe ({optimal_portfolio['sharpe_ratio']:.3f})",
                hovertemplate=f"Optimal Portfolio<br>Return: {optimal_portfolio['expected_annual_return']:.2%}<br>Risk: {optimal_portfolio['annual_volatility']:.2%}<br>Sharpe: {optimal_portfolio['sharpe_ratio']:.3f}<extra></extra>"
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Volatility (Risk)',
            yaxis_title='Expected Return',
            template='plotly_white'
        )
        
        fig.show()
    
    def plot_portfolio_composition(self, 
                                 weights: Union[np.ndarray, Dict],
                                 labels: Optional[List[str]] = None,
                                 title: str = "Portfolio Composition",
                                 chart_type: str = "pie",
                                 figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Plot portfolio composition.
        
        Args:
            weights: Portfolio weights (array or dict)
            labels: Asset labels
            title: Chart title
            chart_type: Chart type ('pie' or 'bar')
            figsize: Figure size
        """
        if isinstance(weights, dict):
            labels = list(weights.keys())
            weights = list(weights.values())
        elif labels is None:
            labels = [f'Asset {i+1}' for i in range(len(weights))]
        
        if chart_type == "pie":
            self._plot_pie_chart(weights, labels, title, figsize)
        elif chart_type == "bar":
            self._plot_bar_chart(weights, labels, title, figsize)
        else:
            raise ValueError("chart_type must be 'pie' or 'bar'")
    
    def _plot_pie_chart(self, 
                       weights: List[float],
                       labels: List[str],
                       title: str,
                       figsize: Tuple[int, int]) -> None:
        """Create pie chart for portfolio composition."""
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = self.color_palette[:len(weights)]
        wedges, texts, autotexts = ax.pie(weights, labels=labels, colors=colors,
                                         autopct='%1.1f%%', startangle=90)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_bar_chart(self, 
                       weights: List[float],
                       labels: List[str],
                       title: str,
                       figsize: Tuple[int, int]) -> None:
        """Create bar chart for portfolio composition."""
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = self.color_palette[:len(weights)]
        bars = ax.bar(labels, weights, color=colors)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_ylabel('Weight', fontsize=12)
        ax.set_xlabel('Assets', fontsize=12)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1%}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_cumulative_returns(self, 
                              returns: Union[pd.Series, pd.DataFrame],
                              benchmark_returns: Optional[pd.Series] = None,
                              title: str = "Cumulative Returns",
                              figsize: Tuple[int, int] = (12, 6),
                              interactive: bool = False,
                              include_excess_returns: bool = False,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None) -> None:
        """
        Plot cumulative returns over time with optional risk-free rate comparison.
        
        Args:
            returns: Portfolio returns (Series) or multiple portfolios (DataFrame)
            benchmark_returns: Benchmark returns for comparison
            title: Chart title
            figsize: Figure size
            interactive: Whether to create interactive chart
            include_excess_returns: Whether to include excess returns over risk-free rate
            start_date: Start date for risk-free rate data
            end_date: End date for risk-free rate data
        """
        # Get risk-free rate if excess returns are requested
        risk_free_rate = None
        if include_excess_returns and start_date and end_date:
            rf_data = self._get_risk_free_rate(start_date, end_date, annualized=False)
            if rf_data is not None:
                # Align risk-free rate with returns data
                if isinstance(returns, pd.Series):
                    aligned_rf = rf_data.reindex(returns.index, method='ffill')
                    risk_free_rate = aligned_rf
                else:
                    aligned_rf = rf_data.reindex(returns.index, method='ffill')
                    risk_free_rate = aligned_rf
        
        if interactive:
            self._plot_cumulative_returns_plotly(returns, benchmark_returns, title, 
                                                risk_free_rate, include_excess_returns)
        else:
            self._plot_cumulative_returns_matplotlib(returns, benchmark_returns, title, figsize,
                                                   risk_free_rate, include_excess_returns)
    
    def _plot_cumulative_returns_matplotlib(self, 
                                          returns: Union[pd.Series, pd.DataFrame],
                                          benchmark_returns: Optional[pd.Series],
                                          title: str,
                                          figsize: Tuple[int, int],
                                          risk_free_rate: Optional[pd.Series] = None,
                                          include_excess_returns: bool = False) -> None:
        """Create matplotlib cumulative returns chart."""
        if include_excess_returns and risk_free_rate is not None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(figsize[0], figsize[1] * 1.5), sharex=True)
        else:
            fig, ax1 = plt.subplots(figsize=figsize)
            ax2 = None
        
        # Plot cumulative returns
        if isinstance(returns, pd.Series):
            cumulative_returns = (1 + returns).cumprod()
            ax1.plot(cumulative_returns.index, cumulative_returns, 
                   label='Portfolio', linewidth=2, color=self.color_palette[0])
            
            # Plot excess returns if requested
            if include_excess_returns and risk_free_rate is not None and ax2 is not None:
                excess_returns = returns - risk_free_rate
                cumulative_excess = (1 + excess_returns).cumprod()
                ax2.plot(cumulative_excess.index, cumulative_excess,
                        label='Portfolio Excess Returns', linewidth=2, color=self.color_palette[0])
        else:
            for i, column in enumerate(returns.columns):
                cumulative_returns = (1 + returns[column]).cumprod()
                ax1.plot(cumulative_returns.index, cumulative_returns, 
                       label=column, linewidth=2, color=self.color_palette[i % len(self.color_palette)])
                
                # Plot excess returns if requested
                if include_excess_returns and risk_free_rate is not None and ax2 is not None:
                    excess_returns = returns[column] - risk_free_rate
                    cumulative_excess = (1 + excess_returns).cumprod()
                    ax2.plot(cumulative_excess.index, cumulative_excess,
                            label=f'{column} Excess', linewidth=2, 
                            color=self.color_palette[i % len(self.color_palette)], linestyle='--')
        
        # Add benchmark if provided
        if benchmark_returns is not None:
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            ax1.plot(benchmark_cumulative.index, benchmark_cumulative, 
                   label='Benchmark', linewidth=2, linestyle='--', color='red')
            
            if include_excess_returns and risk_free_rate is not None and ax2 is not None:
                benchmark_excess = benchmark_returns - risk_free_rate.reindex(benchmark_returns.index, method='ffill')
                benchmark_excess_cumulative = (1 + benchmark_excess).cumprod()
                ax2.plot(benchmark_excess_cumulative.index, benchmark_excess_cumulative,
                        label='Benchmark Excess', linewidth=2, linestyle='--', color='red')
        
        # Add risk-free rate cumulative return if available
        if risk_free_rate is not None:
            rf_cumulative = (1 + risk_free_rate).cumprod()
            ax1.plot(rf_cumulative.index, rf_cumulative,
                   label='Risk-Free Rate', linewidth=1, color='green', alpha=0.7)
        
        ax1.set_title(title, fontsize=16, fontweight='bold')
        ax1.set_ylabel('Cumulative Return', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        if ax2 is not None:
            ax2.set_title('Excess Returns Over Risk-Free Rate', fontweight='bold')
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('Cumulative Excess Return', fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        else:
            ax1.set_xlabel('Date', fontsize=12)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_cumulative_returns_plotly(self, 
                                      returns: Union[pd.Series, pd.DataFrame],
                                      benchmark_returns: Optional[pd.Series],
                                      title: str,
                                      risk_free_rate: Optional[pd.Series] = None,
                                      include_excess_returns: bool = False) -> None:
        """Create interactive plotly cumulative returns chart."""
        if include_excess_returns and risk_free_rate is not None:
            fig = make_subplots(rows=2, cols=1, 
                               subplot_titles=[title, 'Excess Returns Over Risk-Free Rate'],
                               shared_xaxes=True, vertical_spacing=0.1)
        else:
            fig = go.Figure()
        
        # Handle Series or DataFrame
        if isinstance(returns, pd.Series):
            cumulative_returns = (1 + returns).cumprod()
            trace = go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns,
                mode='lines',
                name='Portfolio',
                hovertemplate='Date: %{x}<br>Cumulative Return: %{y:.2%}<extra></extra>'
            )
            
            if include_excess_returns and risk_free_rate is not None:
                fig.add_trace(trace, row=1, col=1)
                
                # Add excess returns
                excess_returns = returns - risk_free_rate
                cumulative_excess = (1 + excess_returns).cumprod()
                fig.add_trace(go.Scatter(
                    x=cumulative_excess.index,
                    y=cumulative_excess,
                    mode='lines',
                    name='Portfolio Excess',
                    hovertemplate='Date: %{x}<br>Cumulative Excess Return: %{y:.2%}<extra></extra>'
                ), row=2, col=1)
            else:
                fig.add_trace(trace)
        else:
            for column in returns.columns:
                cumulative_returns = (1 + returns[column]).cumprod()
                trace = go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns,
                    mode='lines',
                    name=column,
                    hovertemplate=f'{column}<br>Date: %{{x}}<br>Cumulative Return: %{{y:.2%}}<extra></extra>'
                )
                
                if include_excess_returns and risk_free_rate is not None:
                    fig.add_trace(trace, row=1, col=1)
                    
                    # Add excess returns
                    excess_returns = returns[column] - risk_free_rate
                    cumulative_excess = (1 + excess_returns).cumprod()
                    fig.add_trace(go.Scatter(
                        x=cumulative_excess.index,
                        y=cumulative_excess,
                        mode='lines',
                        name=f'{column} Excess',
                        line=dict(dash='dash'),
                        hovertemplate=f'{column} Excess<br>Date: %{{x}}<br>Cumulative Excess Return: %{{y:.2%}}<extra></extra>'
                    ), row=2, col=1)
                else:
                    fig.add_trace(trace)
        
        # Add benchmark if provided
        if benchmark_returns is not None:
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            benchmark_trace = go.Scatter(
                x=benchmark_cumulative.index,
                y=benchmark_cumulative,
                mode='lines',
                name='Benchmark',
                line=dict(dash='dash'),
                hovertemplate='Benchmark<br>Date: %{x}<br>Cumulative Return: %{y:.2%}<extra></extra>'
            )
            
            if include_excess_returns and risk_free_rate is not None:
                fig.add_trace(benchmark_trace, row=1, col=1)
                
                # Add benchmark excess returns
                benchmark_excess = benchmark_returns - risk_free_rate.reindex(benchmark_returns.index, method='ffill')
                benchmark_excess_cumulative = (1 + benchmark_excess).cumprod()
                fig.add_trace(go.Scatter(
                    x=benchmark_excess_cumulative.index,
                    y=benchmark_excess_cumulative,
                    mode='lines',
                    name='Benchmark Excess',
                    line=dict(dash='dash'),
                    hovertemplate='Benchmark Excess<br>Date: %{x}<br>Cumulative Excess Return: %{y:.2%}<extra></extra>'
                ), row=2, col=1)
            else:
                fig.add_trace(benchmark_trace)
        
        # Add risk-free rate if available
        if risk_free_rate is not None:
            rf_cumulative = (1 + risk_free_rate).cumprod()
            rf_trace = go.Scatter(
                x=rf_cumulative.index,
                y=rf_cumulative,
                mode='lines',
                name='Risk-Free Rate',
                line=dict(color='green', width=1),
                opacity=0.7,
                hovertemplate='Risk-Free Rate<br>Date: %{x}<br>Cumulative Return: %{y:.2%}<extra></extra>'
            )
            
            if include_excess_returns:
                fig.add_trace(rf_trace, row=1, col=1)
            else:
                fig.add_trace(rf_trace)
        
        # Update layout
        if include_excess_returns and risk_free_rate is not None:
            fig.update_layout(
                height=800,
                hovermode='x unified',
                template='plotly_white'
            )
            fig.update_xaxes(title_text='Date', row=2, col=1)
            fig.update_yaxes(title_text='Cumulative Return', tickformat='.0%', row=1, col=1)
            fig.update_yaxes(title_text='Cumulative Excess Return', tickformat='.0%', row=2, col=1)
        else:
            fig.update_layout(
                title=title,
                xaxis_title='Date',
                yaxis_title='Cumulative Return',
                yaxis_tickformat='.0%',
                hovermode='x unified',
                template='plotly_white'
            )
        
        fig.show()
    
    def plot_drawdown(self, 
                     returns: pd.Series,
                     title: str = "Portfolio Drawdown",
                     figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot portfolio drawdown over time.
        
        Args:
            returns: Portfolio returns
            title: Chart title
            figsize: Figure size
        """
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Cumulative returns
        ax1.plot(cumulative_returns.index, cumulative_returns, 
                label='Cumulative Returns', color=self.color_palette[0])
        ax1.plot(running_max.index, running_max, 
                label='Peak', color='red', linestyle='--', alpha=0.7)
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title(title, fontweight='bold')
        
        # Drawdown
        ax2.fill_between(drawdown.index, drawdown, 0, 
                        color='red', alpha=0.3, label='Drawdown')
        ax2.plot(drawdown.index, drawdown, color='red')
        ax2.set_ylabel('Drawdown')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        plt.tight_layout()
        plt.show()
    
    def plot_risk_adjusted_metrics(self,
                                 returns: Union[pd.Series, pd.DataFrame],
                                 benchmark_returns: Optional[pd.Series] = None,
                                 start_date: Optional[str] = None,
                                 end_date: Optional[str] = None,
                                 title: str = "Risk-Adjusted Performance Metrics",
                                 figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plot comprehensive risk-adjusted performance metrics.
        
        Args:
            returns: Portfolio returns (Series) or multiple portfolios (DataFrame)
            benchmark_returns: Benchmark returns for comparison
            start_date: Start date for risk-free rate data
            end_date: End date for risk-free rate data
            title: Chart title
            figsize: Figure size
        """
        # Get risk-free rate
        risk_free_rate = None
        if start_date and end_date:
            rf_data = self._get_risk_free_rate(start_date, end_date, annualized=False)
            if rf_data is not None:
                risk_free_rate = rf_data.mean() if hasattr(rf_data, 'mean') else rf_data
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Prepare data
        if isinstance(returns, pd.Series):
            returns_df = pd.DataFrame({'Portfolio': returns})
        else:
            returns_df = returns.copy()
        
        if benchmark_returns is not None:
            returns_df['Benchmark'] = benchmark_returns
        
        # Calculate metrics
        metrics = {}
        for column in returns_df.columns:
            ret_series = returns_df[column].dropna()
            annual_ret = ret_series.mean() * 252
            annual_vol = ret_series.std() * np.sqrt(252)
            
            # Sharpe ratio
            if risk_free_rate is not None:
                excess_ret = annual_ret - risk_free_rate
                sharpe = excess_ret / annual_vol if annual_vol > 0 else 0
            else:
                sharpe = annual_ret / annual_vol if annual_vol > 0 else 0
            
            # Maximum drawdown
            cumulative = (1 + ret_series).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min()
            
            # Calmar ratio
            calmar = annual_ret / abs(max_dd) if max_dd != 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = ret_series[ret_series < 0]
            downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            if risk_free_rate is not None:
                sortino = excess_ret / downside_vol if downside_vol > 0 else 0
            else:
                sortino = annual_ret / downside_vol if downside_vol > 0 else 0
            
            metrics[column] = {
                'Annual Return': annual_ret,
                'Annual Volatility': annual_vol,
                'Sharpe Ratio': sharpe,
                'Max Drawdown': max_dd,
                'Calmar Ratio': calmar,
                'Sortino Ratio': sortino
            }
        
        # Plot 1: Annual Return vs Volatility
        colors = self.color_palette[:len(metrics)]
        for i, (name, data) in enumerate(metrics.items()):
            ax1.scatter(data['Annual Volatility'], data['Annual Return'], 
                       s=100, c=[colors[i]], label=name, alpha=0.7)
            ax1.annotate(name, (data['Annual Volatility'], data['Annual Return']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax1.set_xlabel('Annual Volatility')
        ax1.set_ylabel('Annual Return')
        ax1.set_title('Risk-Return Profile')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Sharpe Ratios
        names = list(metrics.keys())
        sharpe_ratios = [metrics[name]['Sharpe Ratio'] for name in names]
        bars1 = ax2.bar(names, sharpe_ratios, color=colors)
        ax2.set_title('Sharpe Ratios')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, sharpe_ratios):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Plot 3: Maximum Drawdown
        max_drawdowns = [abs(metrics[name]['Max Drawdown']) for name in names]
        bars2 = ax3.bar(names, max_drawdowns, color=colors)
        ax3.set_title('Maximum Drawdown')
        ax3.set_ylabel('Maximum Drawdown (%)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
        # Add value labels on bars
        for bar, value in zip(bars2, max_drawdowns):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{value:.1%}', ha='center', va='bottom')
        
        # Plot 4: Calmar vs Sortino Ratios
        calmar_ratios = [metrics[name]['Calmar Ratio'] for name in names]
        sortino_ratios = [metrics[name]['Sortino Ratio'] for name in names]
        
        x = np.arange(len(names))
        width = 0.35
        
        bars3 = ax4.bar(x - width/2, calmar_ratios, width, label='Calmar Ratio', color=colors, alpha=0.7)
        bars4 = ax4.bar(x + width/2, sortino_ratios, width, label='Sortino Ratio', color=colors, alpha=0.9)
        
        ax4.set_title('Risk-Adjusted Ratios Comparison')
        ax4.set_ylabel('Ratio Value')
        ax4.set_xticks(x)
        ax4.set_xticklabels(names, rotation=45)
        ax4.legend()
        
        # Add value labels on bars
        for bars in [bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def create_dashboard(self, 
                        portfolio_returns: pd.Series,
                        price_data: pd.DataFrame,
                        weights: Dict[str, float],
                        benchmark_returns: Optional[pd.Series] = None,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> None:
        """
        Create a comprehensive portfolio dashboard with risk-free rate integration.
        
        Args:
            portfolio_returns: Portfolio returns series
            price_data: Historical price data
            weights: Portfolio weights
            benchmark_returns: Benchmark returns (optional)
            start_date: Start date for risk-free rate data
            end_date: End date for risk-free rate data
        """
        # Get risk-free rate data
        risk_free_rate = None
        rf_series = None
        if start_date and end_date:
            rf_data = self._get_risk_free_rate(start_date, end_date, annualized=False)
            if rf_data is not None:
                rf_series = rf_data.reindex(portfolio_returns.index, method='ffill')
                risk_free_rate = rf_data.mean() if hasattr(rf_data, 'mean') else rf_data
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=('Price History', 'Portfolio Composition',
                          'Cumulative Returns', 'Excess Returns Over Risk-Free Rate',
                          'Returns Distribution', 'Rolling Sharpe Ratio',
                          'Drawdown', 'Risk-Adjusted Metrics'),
            specs=[[{"secondary_y": False}, {"type": "pie"}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Price history
        for column in price_data.columns:
            fig.add_trace(
                go.Scatter(x=price_data.index, y=price_data[column], 
                          name=column, mode='lines'),
                row=1, col=1
            )
        
        # Portfolio composition
        fig.add_trace(
            go.Pie(labels=list(weights.keys()), values=list(weights.values()),
                   name="Portfolio Weights"),
            row=1, col=2
        )
        
        # Cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        fig.add_trace(
            go.Scatter(x=cumulative_returns.index, y=cumulative_returns,
                      name='Portfolio', mode='lines'),
            row=2, col=1
        )
        
        if benchmark_returns is not None:
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            fig.add_trace(
                go.Scatter(x=benchmark_cumulative.index, y=benchmark_cumulative,
                          name='Benchmark', mode='lines', line=dict(dash='dash')),
                row=2, col=1
            )
        
        # Add risk-free rate cumulative return
        if rf_series is not None:
            rf_cumulative = (1 + rf_series).cumprod()
            fig.add_trace(
                go.Scatter(x=rf_cumulative.index, y=rf_cumulative,
                          name='Risk-Free Rate', mode='lines', 
                          line=dict(color='green', width=1), opacity=0.7),
                row=2, col=1
            )
        
        # Excess returns over risk-free rate
        if rf_series is not None:
            excess_returns = portfolio_returns - rf_series
            excess_cumulative = (1 + excess_returns).cumprod()
            fig.add_trace(
                go.Scatter(x=excess_cumulative.index, y=excess_cumulative,
                          name='Portfolio Excess', mode='lines'),
                row=2, col=2
            )
            
            if benchmark_returns is not None:
                benchmark_excess = benchmark_returns - rf_series.reindex(benchmark_returns.index, method='ffill')
                benchmark_excess_cumulative = (1 + benchmark_excess).cumprod()
                fig.add_trace(
                    go.Scatter(x=benchmark_excess_cumulative.index, y=benchmark_excess_cumulative,
                              name='Benchmark Excess', mode='lines', line=dict(dash='dash')),
                    row=2, col=2
                )
        else:
            # If no risk-free rate, show regular returns distribution
            fig.add_trace(
                go.Histogram(x=portfolio_returns, name='Returns Distribution',
                            nbinsx=50, opacity=0.7),
                row=2, col=2
            )
        
        # Returns distribution
        fig.add_trace(
            go.Histogram(x=portfolio_returns, name='Portfolio Returns',
                        nbinsx=50, opacity=0.7),
            row=3, col=1
        )
        
        # Rolling Sharpe ratio (1-year window)
        if risk_free_rate is not None:
            rolling_excess = portfolio_returns.rolling(252).mean() * 252 - risk_free_rate
            rolling_vol = portfolio_returns.rolling(252).std() * np.sqrt(252)
            rolling_sharpe = rolling_excess / rolling_vol
        else:
            rolling_sharpe = (portfolio_returns.rolling(252).mean() * 252) / (portfolio_returns.rolling(252).std() * np.sqrt(252))
        
        fig.add_trace(
            go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe,
                      name='Rolling Sharpe (1Y)', mode='lines'),
            row=3, col=2
        )
        
        # Drawdown
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown, 
                      fill='tonexty', name='Drawdown'),
            row=4, col=1
        )
        
        # Risk-adjusted metrics comparison
        if benchmark_returns is not None:
            # Calculate metrics for both portfolio and benchmark
            portfolio_annual_ret = portfolio_returns.mean() * 252
            portfolio_annual_vol = portfolio_returns.std() * np.sqrt(252)
            benchmark_annual_ret = benchmark_returns.mean() * 252
            benchmark_annual_vol = benchmark_returns.std() * np.sqrt(252)
            
            if risk_free_rate is not None:
                portfolio_sharpe = (portfolio_annual_ret - risk_free_rate) / portfolio_annual_vol
                benchmark_sharpe = (benchmark_annual_ret - risk_free_rate) / benchmark_annual_vol
            else:
                portfolio_sharpe = portfolio_annual_ret / portfolio_annual_vol
                benchmark_sharpe = benchmark_annual_ret / benchmark_annual_vol
            
            metrics_names = ['Portfolio', 'Benchmark']
            sharpe_values = [portfolio_sharpe, benchmark_sharpe]
            
            fig.add_trace(
                go.Bar(x=metrics_names, y=sharpe_values, name='Sharpe Ratios'),
                row=4, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text=f"Portfolio Analytics Dashboard{' (with Risk-Free Rate)' if risk_free_rate else ''}",
            title_x=0.5,
            height=1600,
            showlegend=False,
            template='plotly_white'
        )
        
        # Update y-axis formats
        fig.update_yaxes(tickformat='.0%', row=2, col=1)  # Cumulative returns
        fig.update_yaxes(tickformat='.0%', row=2, col=2)  # Excess returns
        fig.update_yaxes(tickformat='.0%', row=4, col=1)  # Drawdown
        
        fig.show()
