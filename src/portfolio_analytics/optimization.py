"""
Portfolio optimization algorithms and methods.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional, Tuple, Dict, List
from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    """Base class for portfolio optimizers."""
    
    @abstractmethod
    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        """Optimize portfolio weights."""
        pass


class PortfolioOptimizer(BaseOptimizer):
    """
    Portfolio optimization using modern portfolio theory and other methods.
    
    Supports various optimization objectives including:
    - Maximum Sharpe ratio
    - Minimum variance
    - Target return with minimum risk
    - Maximum return for given risk level
    """
    
    def __init__(self):
        """Initialize the portfolio optimizer."""
        self.last_optimization_result = None
    
    def optimize(self, 
                returns: pd.DataFrame,
                method: str = "max_sharpe",
                risk_free_rate: float = 0.02,
                target_return: Optional[float] = None,
                target_risk: Optional[float] = None,
                constraints: Optional[List[Dict]] = None,
                bounds: Optional[Tuple] = None) -> np.ndarray:
        """
        Optimize portfolio weights based on the specified method.
        
        Args:
            returns: DataFrame of asset returns
            method: Optimization method ('max_sharpe', 'min_variance', 'target_return', 'max_return')
            risk_free_rate: Risk-free rate (annualized)
            target_return: Target return for constrained optimization
            target_risk: Target risk level for constrained optimization
            constraints: Additional constraints for optimization
            bounds: Bounds for individual weights (min, max)
        
        Returns:
            Array of optimized weights
        """
        n_assets = len(returns.columns)
        
        # Default bounds: no short selling (0, 1)
        if bounds is None:
            bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Default constraints: weights sum to 1
        default_constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        if constraints:
            default_constraints.extend(constraints)
        
        # Initial guess: equal weights
        initial_guess = np.array([1.0 / n_assets] * n_assets)
        
        # Calculate expected returns and covariance matrix
        expected_returns = returns.mean() * 252  # Annualized
        cov_matrix = returns.cov() * 252  # Annualized
        
        if method == "max_sharpe":
            result = self._maximize_sharpe_ratio(
                expected_returns, cov_matrix, risk_free_rate, 
                initial_guess, bounds, default_constraints
            )
        elif method == "min_variance":
            result = self._minimize_variance(
                cov_matrix, initial_guess, bounds, default_constraints
            )
        elif method == "target_return":
            if target_return is None:
                raise ValueError("target_return must be specified for target_return optimization")
            result = self._target_return_optimization(
                expected_returns, cov_matrix, target_return,
                initial_guess, bounds, default_constraints
            )
        elif method == "max_return":
            if target_risk is None:
                raise ValueError("target_risk must be specified for max_return optimization")
            result = self._maximize_return_for_risk(
                expected_returns, cov_matrix, target_risk,
                initial_guess, bounds, default_constraints
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        self.last_optimization_result = result
        return result.x
    
    def _maximize_sharpe_ratio(self, 
                              expected_returns: pd.Series,
                              cov_matrix: pd.DataFrame,
                              risk_free_rate: float,
                              initial_guess: np.ndarray,
                              bounds: Tuple,
                              constraints: List[Dict]) -> object:
        """Maximize Sharpe ratio optimization."""
        
        def negative_sharpe_ratio(weights):
            portfolio_return = np.sum(expected_returns * weights)
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            if portfolio_volatility == 0:
                return -np.inf
            
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
            return -sharpe_ratio  # Negative because we want to maximize
        
        result = minimize(
            negative_sharpe_ratio,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'disp': False}
        )
        
        return result
    
    def _minimize_variance(self, 
                          cov_matrix: pd.DataFrame,
                          initial_guess: np.ndarray,
                          bounds: Tuple,
                          constraints: List[Dict]) -> object:
        """Minimum variance optimization."""
        
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        result = minimize(
            portfolio_variance,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'disp': False}
        )
        
        return result
    
    def _target_return_optimization(self, 
                                   expected_returns: pd.Series,
                                   cov_matrix: pd.DataFrame,
                                   target_return: float,
                                   initial_guess: np.ndarray,
                                   bounds: Tuple,
                                   constraints: List[Dict]) -> object:
        """Minimize risk for target return."""
        
        # Add target return constraint
        return_constraint = {
            'type': 'eq',
            'fun': lambda x: np.sum(expected_returns * x) - target_return
        }
        constraints.append(return_constraint)
        
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        result = minimize(
            portfolio_variance,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'disp': False}
        )
        
        return result
    
    def _maximize_return_for_risk(self, 
                                 expected_returns: pd.Series,
                                 cov_matrix: pd.DataFrame,
                                 target_risk: float,
                                 initial_guess: np.ndarray,
                                 bounds: Tuple,
                                 constraints: List[Dict]) -> object:
        """Maximize return for target risk level."""
        
        # Add risk constraint
        risk_constraint = {
            'type': 'eq',
            'fun': lambda x: np.sqrt(np.dot(x.T, np.dot(cov_matrix, x))) - target_risk
        }
        constraints.append(risk_constraint)
        
        def negative_portfolio_return(weights):
            return -np.sum(expected_returns * weights)
        
        result = minimize(
            negative_portfolio_return,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'disp': False}
        )
        
        return result
    
    def calculate_efficient_frontier(self, 
                                   returns: pd.DataFrame,
                                   num_portfolios: int = 100,
                                   risk_free_rate: float = 0.02) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the efficient frontier.
        
        Args:
            returns: DataFrame of asset returns
            num_portfolios: Number of portfolios to calculate
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        
        Returns:
            Tuple of (returns_array, volatilities_array, sharpe_ratios_array)
        """
        expected_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        # Calculate range of target returns
        min_ret = expected_returns.min()
        max_ret = expected_returns.max()
        target_returns = np.linspace(min_ret, max_ret, num_portfolios)
        
        efficient_portfolios = []
        
        for target_ret in target_returns:
            try:
                weights = self.optimize(
                    returns=returns,
                    method="target_return",
                    target_return=target_ret
                )
                
                portfolio_return = np.sum(expected_returns * weights)
                portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                portfolio_volatility = np.sqrt(portfolio_variance)
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
                
                efficient_portfolios.append({
                    'return': portfolio_return,
                    'volatility': portfolio_volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'weights': weights
                })
            except:
                # Skip if optimization fails for this target return
                continue
        
        if not efficient_portfolios:
            raise ValueError("Could not calculate efficient frontier")
        
        returns_array = np.array([p['return'] for p in efficient_portfolios])
        volatilities_array = np.array([p['volatility'] for p in efficient_portfolios])
        sharpe_ratios_array = np.array([p['sharpe_ratio'] for p in efficient_portfolios])
        
        return returns_array, volatilities_array, sharpe_ratios_array
    
    def get_optimization_summary(self, 
                               returns: pd.DataFrame,
                               weights: np.ndarray,
                               risk_free_rate: float = 0.02) -> Dict:
        """
        Get summary of optimized portfolio.
        
        Args:
            returns: DataFrame of asset returns
            weights: Optimized weights
            risk_free_rate: Risk-free rate
        
        Returns:
            Dictionary with portfolio metrics
        """
        expected_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        portfolio_return = np.sum(expected_returns * weights)
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        
        return {
            'expected_annual_return': portfolio_return,
            'annual_volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'weights': dict(zip(returns.columns, weights)),
            'diversification_ratio': self._calculate_diversification_ratio(weights, cov_matrix),
            'maximum_weight': weights.max(),
            'minimum_weight': weights.min(),
            'effective_number_of_assets': 1 / np.sum(weights**2)
        }
    
    def _calculate_diversification_ratio(self, 
                                       weights: np.ndarray,
                                       cov_matrix: pd.DataFrame) -> float:
        """Calculate diversification ratio."""
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        weighted_average_volatility = np.sum(weights * np.sqrt(np.diag(cov_matrix)))
        
        if portfolio_volatility == 0:
            return 0
        
        return weighted_average_volatility / portfolio_volatility


class BlackLittermanOptimizer(BaseOptimizer):
    """
    Black-Litterman portfolio optimization.
    
    Combines market equilibrium with investor views to generate
    expected returns and optimize portfolio weights.
    """
    
    def __init__(self, risk_aversion: float = 3.0):
        """
        Initialize Black-Litterman optimizer.
        
        Args:
            risk_aversion: Risk aversion parameter (typically 2-5)
        """
        self.risk_aversion = risk_aversion
    
    def optimize(self, 
                returns: pd.DataFrame,
                market_caps: Optional[Dict[str, float]] = None,
                views: Optional[Dict] = None,
                **kwargs) -> np.ndarray:
        """
        Optimize using Black-Litterman model.
        
        Args:
            returns: DataFrame of asset returns
            market_caps: Dictionary of market capitalizations
            views: Dictionary with investor views
            **kwargs: Additional arguments
        
        Returns:
            Array of optimized weights
        """
        # This is a simplified implementation
        # In practice, you would implement the full Black-Litterman model
        
        if market_caps is None:
            # Equal market caps assumption
            market_caps = {asset: 1.0 for asset in returns.columns}
        
        # Calculate market portfolio weights
        total_market_cap = sum(market_caps.values())
        market_weights = np.array([market_caps[asset] / total_market_cap 
                                 for asset in returns.columns])
        
        # For now, return market weights (can be extended with views)
        return market_weights
