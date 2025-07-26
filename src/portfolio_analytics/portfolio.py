"""
Core Portfolio class for managing and analyzing investment portfolios.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta

from .data_provider import DataProvider
from .optimization import PortfolioOptimizer
from .performance import PerformanceAnalyzer
from .risk_models import RiskModel


class Portfolio:
    """
    A comprehensive portfolio management and analysis class.
    
    This class provides functionality for portfolio construction, optimization,
    performance analysis, and risk management.
    """
    
    def __init__(self, 
                 symbols: List[str], 
                 weights: Optional[List[float]] = None,
                 name: str = "Portfolio"):
        """
        Initialize a portfolio with given symbols and weights.
        
        Args:
            symbols: List of stock symbols (e.g., ['AAPL', 'GOOGL', 'MSFT'])
            weights: List of portfolio weights (must sum to 1.0)
            name: Portfolio name for identification
        """
        self.symbols = symbols
        self.name = name
        self.data: Optional[pd.DataFrame] = None
        self.returns: Optional[pd.DataFrame] = None
        
        # Set equal weights if none provided
        if weights is None:
            self.weights = np.array([1.0 / len(symbols)] * len(symbols))
        else:
            if len(weights) != len(symbols):
                raise ValueError("Number of weights must match number of symbols")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.0")
            self.weights = np.array(weights)
            
        # Initialize analysis components
        self.optimizer = PortfolioOptimizer()
        self.performance_analyzer = PerformanceAnalyzer()
        self.risk_model = RiskModel()
    
    def load_data(self, 
                  data_provider: DataProvider,
                  start_date: str = "2020-01-01", 
                  end_date: Optional[str] = None) -> None:
        """
        Load historical price data for portfolio symbols.
        
        Args:
            data_provider: DataProvider instance for fetching data
            start_date: Start date for historical data (YYYY-MM-DD)
            end_date: End date for historical data (YYYY-MM-DD)
        """
        self.data = data_provider.get_price_data(
            symbols=self.symbols,
            start_date=start_date,
            end_date=end_date
        )
        
        # Calculate returns
        self.returns = self.data.pct_change().dropna()
    
    def portfolio_return(self) -> float:
        """Calculate portfolio return based on current weights."""
        if self.returns is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        weighted_returns = (self.returns * self.weights).sum(axis=1)
        return weighted_returns.mean()
    
    def annual_return(self) -> float:
        """Calculate annualized portfolio return."""
        return self.portfolio_return() * 252  # 252 trading days per year
    
    def portfolio_volatility(self) -> float:
        """Calculate portfolio volatility (standard deviation of returns)."""
        if self.returns is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        portfolio_returns = (self.returns * self.weights).sum(axis=1)
        return portfolio_returns.std()
    
    def annual_volatility(self) -> float:
        """Calculate annualized portfolio volatility."""
        return self.portfolio_volatility() * np.sqrt(252)
    
    def sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            risk_free_rate: Risk-free rate (annual)
        
        Returns:
            Sharpe ratio
        """
        annual_ret = self.annual_return()
        annual_vol = self.annual_volatility()
        
        if annual_vol == 0:
            return 0
        
        return (annual_ret - risk_free_rate) / annual_vol
    
    def optimize(self, 
                 method: str = "max_sharpe",
                 risk_free_rate: float = 0.02,
                 target_return: Optional[float] = None) -> np.ndarray:
        """
        Optimize portfolio weights.
        
        Args:
            method: Optimization method ('max_sharpe', 'min_variance', 'target_return')
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            target_return: Target return for constrained optimization
        
        Returns:
            Optimized weights array
        """
        if self.returns is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        optimized_weights = self.optimizer.optimize(
            returns=self.returns,
            method=method,
            risk_free_rate=risk_free_rate,
            target_return=target_return
        )
        
        # Update portfolio weights
        self.weights = optimized_weights
        return optimized_weights
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get comprehensive performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        if self.returns is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        portfolio_returns = (self.returns * self.weights).sum(axis=1)
        
        return self.performance_analyzer.calculate_metrics(
            returns=portfolio_returns,
            benchmark_returns=None  # TODO: Add benchmark comparison
        )
    
    def calculate_var(self, 
                     confidence_level: float = 0.95,
                     method: str = "historical") -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            method: VaR calculation method ('historical', 'parametric', 'monte_carlo')
        
        Returns:
            VaR value
        """
        if self.returns is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        portfolio_returns = (self.returns * self.weights).sum(axis=1)
        
        return self.risk_model.calculate_var(
            returns=portfolio_returns,
            confidence_level=confidence_level,
            method=method
        )
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """Get correlation matrix of portfolio assets."""
        if self.returns is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        return self.returns.corr()
    
    def summary(self) -> Dict[str, any]:
        """
        Get portfolio summary with key metrics.
        
        Returns:
            Dictionary containing portfolio summary
        """
        if self.data is None:
            return {
                "name": self.name,
                "symbols": self.symbols,
                "weights": self.weights.tolist(),
                "status": "No data loaded"
            }
        
        return {
            "name": self.name,
            "symbols": self.symbols,
            "weights": self.weights.tolist(),
            "annual_return": self.annual_return(),
            "annual_volatility": self.annual_volatility(),
            "sharpe_ratio": self.sharpe_ratio(),
            "var_95": self.calculate_var(0.95),
            "data_period": {
                "start": self.data.index[0].strftime("%Y-%m-%d"),
                "end": self.data.index[-1].strftime("%Y-%m-%d"),
                "days": len(self.data)
            }
        }
    
    def __repr__(self) -> str:
        return f"Portfolio(name='{self.name}', symbols={self.symbols}, n_assets={len(self.symbols)})"
