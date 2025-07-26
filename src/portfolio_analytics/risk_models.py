"""
Risk models and Value at Risk (VaR) calculations.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Dict, List, Tuple
from abc import ABC, abstractmethod


class BaseRiskModel(ABC):
    """Base class for risk models."""
    
    @abstractmethod
    def calculate_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Value at Risk."""
        pass


class RiskModel(BaseRiskModel):
    """
    Comprehensive risk model for portfolio analysis.
    
    Provides various risk metrics including VaR, Expected Shortfall,
    maximum drawdown, and other risk measures.
    """
    
    def __init__(self):
        """Initialize the risk model."""
        pass
    
    def calculate_var(self, 
                     returns: pd.Series,
                     confidence_level: float = 0.95,
                     method: str = "historical") -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Series of portfolio returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            method: VaR calculation method ('historical', 'parametric', 'monte_carlo')
        
        Returns:
            VaR value (positive number representing loss)
        """
        if method == "historical":
            return self._historical_var(returns, confidence_level)
        elif method == "parametric":
            return self._parametric_var(returns, confidence_level)
        elif method == "monte_carlo":
            return self._monte_carlo_var(returns, confidence_level)
        else:
            raise ValueError(f"Unknown VaR method: {method}")
    
    def _historical_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate historical VaR."""
        if len(returns) == 0:
            return 0.0
        
        # Sort returns in ascending order
        sorted_returns = returns.sort_values()
        
        # Find the percentile corresponding to the confidence level
        percentile = (1 - confidence_level) * 100
        var_value = np.percentile(sorted_returns, percentile)
        
        # Return positive value (loss)
        return -var_value
    
    def _parametric_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate parametric VaR assuming normal distribution."""
        if len(returns) == 0:
            return 0.0
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Z-score for the confidence level
        z_score = stats.norm.ppf(1 - confidence_level)
        
        # VaR calculation
        var_value = mean_return + z_score * std_return
        
        # Return positive value (loss)
        return -var_value
    
    def _monte_carlo_var(self, 
                        returns: pd.Series, 
                        confidence_level: float,
                        num_simulations: int = 10000) -> float:
        """Calculate Monte Carlo VaR."""
        if len(returns) == 0:
            return 0.0
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Generate random returns
        simulated_returns = np.random.normal(mean_return, std_return, num_simulations)
        
        # Calculate VaR from simulated returns
        percentile = (1 - confidence_level) * 100
        var_value = np.percentile(simulated_returns, percentile)
        
        # Return positive value (loss)
        return -var_value
    
    def calculate_expected_shortfall(self, 
                                   returns: pd.Series,
                                   confidence_level: float = 0.95,
                                   method: str = "historical") -> float:
        """
        Calculate Expected Shortfall (Conditional VaR).
        
        Args:
            returns: Series of portfolio returns
            confidence_level: Confidence level
            method: Calculation method
        
        Returns:
            Expected Shortfall value
        """
        var_value = self.calculate_var(returns, confidence_level, method)
        
        if method == "historical":
            # Find returns below VaR threshold
            threshold = -var_value  # Convert back to negative for comparison
            tail_returns = returns[returns <= threshold]
            
            if len(tail_returns) == 0:
                return var_value
            
            expected_shortfall = -tail_returns.mean()
            return expected_shortfall
        
        elif method == "parametric":
            # Parametric ES for normal distribution
            mean_return = returns.mean()
            std_return = returns.std()
            
            z_score = stats.norm.ppf(1 - confidence_level)
            density = stats.norm.pdf(z_score)
            
            es_value = mean_return - std_return * density / (1 - confidence_level)
            return -es_value
        
        else:
            # For other methods, approximate using historical approach
            return self.calculate_expected_shortfall(returns, confidence_level, "historical")
    
    def calculate_maximum_drawdown(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate maximum drawdown and related metrics.
        
        Args:
            returns: Series of portfolio returns
        
        Returns:
            Dictionary with drawdown metrics
        """
        if len(returns) == 0:
            return {'max_drawdown': 0.0, 'drawdown_duration': 0, 'recovery_time': 0}
        
        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cumulative_returns.expanding().max()
        
        # Calculate drawdown
        drawdown = (cumulative_returns - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = drawdown.min()
        
        # Find the period of maximum drawdown
        max_dd_date = drawdown.idxmin()
        
        # Calculate drawdown duration and recovery time
        drawdown_start = running_max[running_max.index <= max_dd_date].idxmax()
        
        # Find recovery date (when cumulative return exceeds previous peak)
        recovery_dates = cumulative_returns[cumulative_returns.index > max_dd_date]
        peak_value = running_max.loc[max_dd_date]
        recovery_mask = recovery_dates >= peak_value
        
        if recovery_mask.any():
            recovery_date = recovery_dates[recovery_mask].index[0]
            recovery_time = len(returns[max_dd_date:recovery_date]) - 1
        else:
            recovery_time = len(returns[max_dd_date:]) - 1  # Still in drawdown
        
        drawdown_duration = len(returns[drawdown_start:max_dd_date])
        
        return {
            'max_drawdown': abs(max_drawdown),
            'drawdown_duration': drawdown_duration,
            'recovery_time': recovery_time,
            'max_drawdown_date': max_dd_date
        }
    
    def calculate_downside_deviation(self, 
                                   returns: pd.Series,
                                   target_return: float = 0.0) -> float:
        """
        Calculate downside deviation.
        
        Args:
            returns: Series of portfolio returns
            target_return: Target return threshold
        
        Returns:
            Downside deviation
        """
        if len(returns) == 0:
            return 0.0
        
        # Only consider returns below target
        downside_returns = returns[returns < target_return]
        
        if len(downside_returns) == 0:
            return 0.0
        
        # Calculate downside deviation
        downside_deviation = np.sqrt(np.mean((downside_returns - target_return) ** 2))
        
        return downside_deviation
    
    def calculate_sortino_ratio(self, 
                              returns: pd.Series,
                              risk_free_rate: float = 0.02,
                              target_return: Optional[float] = None) -> float:
        """
        Calculate Sortino ratio.
        
        Args:
            returns: Series of portfolio returns
            risk_free_rate: Risk-free rate (annualized)
            target_return: Target return (defaults to risk-free rate)
        
        Returns:
            Sortino ratio
        """
        if target_return is None:
            target_return = risk_free_rate / 252  # Convert to daily
        
        annual_return = returns.mean() * 252
        downside_deviation = self.calculate_downside_deviation(returns, target_return)
        
        if downside_deviation == 0:
            return np.inf if annual_return > risk_free_rate else 0
        
        sortino_ratio = (annual_return - risk_free_rate) / (downside_deviation * np.sqrt(252))
        
        return sortino_ratio
    
    def calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """
        Calculate Calmar ratio (annual return / maximum drawdown).
        
        Args:
            returns: Series of portfolio returns
        
        Returns:
            Calmar ratio
        """
        annual_return = returns.mean() * 252
        max_drawdown_info = self.calculate_maximum_drawdown(returns)
        max_drawdown = max_drawdown_info['max_drawdown']
        
        if max_drawdown == 0:
            return np.inf if annual_return > 0 else 0
        
        calmar_ratio = annual_return / max_drawdown
        
        return calmar_ratio
    
    def calculate_risk_metrics(self, 
                             returns: pd.Series,
                             confidence_levels: List[float] = [0.95, 0.99],
                             risk_free_rate: float = 0.02) -> Dict[str, any]:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            returns: Series of portfolio returns
            confidence_levels: List of confidence levels for VaR/ES
            risk_free_rate: Risk-free rate
        
        Returns:
            Dictionary with all risk metrics
        """
        metrics = {}
        
        # Basic statistics
        metrics['volatility'] = returns.std() * np.sqrt(252)
        metrics['skewness'] = returns.skew()
        metrics['kurtosis'] = returns.kurtosis()
        
        # VaR and Expected Shortfall for different confidence levels
        for confidence in confidence_levels:
            conf_str = f"{int(confidence * 100)}"
            metrics[f'var_{conf_str}'] = self.calculate_var(returns, confidence)
            metrics[f'expected_shortfall_{conf_str}'] = self.calculate_expected_shortfall(returns, confidence)
        
        # Drawdown metrics
        drawdown_metrics = self.calculate_maximum_drawdown(returns)
        metrics.update(drawdown_metrics)
        
        # Downside risk metrics
        metrics['downside_deviation'] = self.calculate_downside_deviation(returns)
        metrics['sortino_ratio'] = self.calculate_sortino_ratio(returns, risk_free_rate)
        metrics['calmar_ratio'] = self.calculate_calmar_ratio(returns)
        
        # Additional risk measures
        metrics['tracking_error'] = returns.std() * np.sqrt(252)  # Assuming benchmark return is 0
        
        return metrics


class VaRCalculator:
    """Specialized class for Value at Risk calculations."""
    
    def __init__(self):
        """Initialize VaR calculator."""
        self.risk_model = RiskModel()
    
    def calculate_portfolio_var(self, 
                              portfolio_returns: pd.Series,
                              confidence_level: float = 0.95,
                              method: str = "historical") -> float:
        """Calculate portfolio VaR."""
        return self.risk_model.calculate_var(portfolio_returns, confidence_level, method)
    
    def calculate_component_var(self, 
                              returns: pd.DataFrame,
                              weights: np.ndarray,
                              confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Calculate component VaR for each asset.
        
        Args:
            returns: DataFrame of asset returns
            weights: Portfolio weights
            confidence_level: Confidence level
        
        Returns:
            Dictionary of component VaR values
        """
        portfolio_returns = (returns * weights).sum(axis=1)
        portfolio_var = self.calculate_portfolio_var(portfolio_returns, confidence_level)
        
        component_vars = {}
        
        for i, asset in enumerate(returns.columns):
            # Calculate marginal VaR by changing weight slightly
            epsilon = 0.001
            new_weights = weights.copy()
            new_weights[i] += epsilon
            new_weights = new_weights / new_weights.sum()  # Renormalize
            
            new_portfolio_returns = (returns * new_weights).sum(axis=1)
            new_var = self.calculate_portfolio_var(new_portfolio_returns, confidence_level)
            
            marginal_var = (new_var - portfolio_var) / epsilon
            component_var = marginal_var * weights[i]
            
            component_vars[asset] = component_var
        
        return component_vars
