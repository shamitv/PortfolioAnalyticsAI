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


class PortfolioVisualizer:
    """
    Comprehensive visualization tools for portfolio analysis.
    
    Provides static and interactive charts for portfolio performance,
    risk analysis, and optimization results.
    """
    
    def __init__(self, style: str = "seaborn"):
        """
        Initialize the visualizer.
        
        Args:
            style: Matplotlib style ('seaborn', 'ggplot', 'bmh', etc.)
        """
        plt.style.use(style if style in plt.style.available else 'default')
        self.color_palette = sns.color_palette("husl", 10)
    
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
                              interactive: bool = False) -> None:
        """
        Plot efficient frontier.
        
        Args:
            returns_array: Array of portfolio returns
            volatilities_array: Array of portfolio volatilities
            sharpe_ratios_array: Array of Sharpe ratios
            optimal_portfolio: Dictionary with optimal portfolio metrics
            title: Chart title
            figsize: Figure size
            interactive: Whether to create interactive chart
        """
        if interactive:
            self._plot_efficient_frontier_plotly(
                returns_array, volatilities_array, sharpe_ratios_array, 
                optimal_portfolio, title
            )
        else:
            self._plot_efficient_frontier_matplotlib(
                returns_array, volatilities_array, sharpe_ratios_array,
                optimal_portfolio, title, figsize
            )
    
    def _plot_efficient_frontier_matplotlib(self, 
                                          returns_array: np.ndarray,
                                          volatilities_array: np.ndarray,
                                          sharpe_ratios_array: np.ndarray,
                                          optimal_portfolio: Optional[Dict],
                                          title: str,
                                          figsize: Tuple[int, int]) -> None:
        """Create matplotlib efficient frontier chart."""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot efficient frontier
        scatter = ax.scatter(volatilities_array, returns_array, 
                           c=sharpe_ratios_array, cmap='viridis', 
                           alpha=0.6, s=50)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Sharpe Ratio', rotation=270, labelpad=20)
        
        # Highlight optimal portfolio if provided
        if optimal_portfolio:
            ax.scatter(optimal_portfolio['annual_volatility'], 
                      optimal_portfolio['expected_annual_return'],
                      marker='*', color='red', s=500, 
                      label=f"Max Sharpe Ratio: {optimal_portfolio['sharpe_ratio']:.3f}")
            ax.legend()
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Volatility (Risk)', fontsize=12)
        ax.set_ylabel('Expected Return', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_efficient_frontier_plotly(self, 
                                      returns_array: np.ndarray,
                                      volatilities_array: np.ndarray,
                                      sharpe_ratios_array: np.ndarray,
                                      optimal_portfolio: Optional[Dict],
                                      title: str) -> None:
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
                              interactive: bool = False) -> None:
        """
        Plot cumulative returns over time.
        
        Args:
            returns: Portfolio returns (Series) or multiple portfolios (DataFrame)
            benchmark_returns: Benchmark returns for comparison
            title: Chart title
            figsize: Figure size
            interactive: Whether to create interactive chart
        """
        if interactive:
            self._plot_cumulative_returns_plotly(returns, benchmark_returns, title)
        else:
            self._plot_cumulative_returns_matplotlib(returns, benchmark_returns, title, figsize)
    
    def _plot_cumulative_returns_matplotlib(self, 
                                          returns: Union[pd.Series, pd.DataFrame],
                                          benchmark_returns: Optional[pd.Series],
                                          title: str,
                                          figsize: Tuple[int, int]) -> None:
        """Create matplotlib cumulative returns chart."""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Handle Series or DataFrame
        if isinstance(returns, pd.Series):
            cumulative_returns = (1 + returns).cumprod()
            ax.plot(cumulative_returns.index, cumulative_returns, 
                   label='Portfolio', linewidth=2, color=self.color_palette[0])
        else:
            for i, column in enumerate(returns.columns):
                cumulative_returns = (1 + returns[column]).cumprod()
                ax.plot(cumulative_returns.index, cumulative_returns, 
                       label=column, linewidth=2, color=self.color_palette[i % len(self.color_palette)])
        
        # Add benchmark if provided
        if benchmark_returns is not None:
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            ax.plot(benchmark_cumulative.index, benchmark_cumulative, 
                   label='Benchmark', linewidth=2, linestyle='--', color='red')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Return', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        plt.tight_layout()
        plt.show()
    
    def _plot_cumulative_returns_plotly(self, 
                                      returns: Union[pd.Series, pd.DataFrame],
                                      benchmark_returns: Optional[pd.Series],
                                      title: str) -> None:
        """Create interactive plotly cumulative returns chart."""
        fig = go.Figure()
        
        # Handle Series or DataFrame
        if isinstance(returns, pd.Series):
            cumulative_returns = (1 + returns).cumprod()
            fig.add_trace(go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns,
                mode='lines',
                name='Portfolio',
                hovertemplate='Date: %{x}<br>Cumulative Return: %{y:.2%}<extra></extra>'
            ))
        else:
            for column in returns.columns:
                cumulative_returns = (1 + returns[column]).cumprod()
                fig.add_trace(go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns,
                    mode='lines',
                    name=column,
                    hovertemplate=f'{column}<br>Date: %{{x}}<br>Cumulative Return: %{{y:.2%}}<extra></extra>'
                ))
        
        # Add benchmark if provided
        if benchmark_returns is not None:
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            fig.add_trace(go.Scatter(
                x=benchmark_cumulative.index,
                y=benchmark_cumulative,
                mode='lines',
                name='Benchmark',
                line=dict(dash='dash'),
                hovertemplate='Benchmark<br>Date: %{x}<br>Cumulative Return: %{y:.2%}<extra></extra>'
            ))
        
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
    
    def create_dashboard(self, 
                        portfolio_returns: pd.Series,
                        price_data: pd.DataFrame,
                        weights: Dict[str, float],
                        benchmark_returns: Optional[pd.Series] = None) -> None:
        """
        Create a comprehensive portfolio dashboard.
        
        Args:
            portfolio_returns: Portfolio returns series
            price_data: Historical price data
            weights: Portfolio weights
            benchmark_returns: Benchmark returns (optional)
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Price History', 'Portfolio Composition',
                          'Cumulative Returns', 'Returns Distribution',
                          'Drawdown', 'Rolling Metrics'),
            specs=[[{"secondary_y": False}, {"type": "pie"}],
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
        
        # Returns distribution
        fig.add_trace(
            go.Histogram(x=portfolio_returns, name='Returns Distribution',
                        nbinsx=50, opacity=0.7),
            row=2, col=2
        )
        
        # Drawdown
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown, 
                      fill='tonexty', name='Drawdown'),
            row=3, col=1
        )
        
        # Rolling Sharpe ratio
        rolling_sharpe = (portfolio_returns.rolling(252).mean() * 252) / (portfolio_returns.rolling(252).std() * np.sqrt(252))
        fig.add_trace(
            go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe,
                      name='Rolling Sharpe (1Y)', mode='lines'),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Portfolio Analytics Dashboard",
            title_x=0.5,
            height=1200,
            showlegend=False,
            template='plotly_white'
        )
        
        fig.show()
