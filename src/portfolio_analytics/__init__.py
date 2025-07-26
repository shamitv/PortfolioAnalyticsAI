"""
Portfolio Analytics AI

A comprehensive Python package for AI-powered portfolio analytics, 
optimization, and risk management.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .portfolio import Portfolio
from .data_provider import DataProvider
from .optimization import PortfolioOptimizer
from .risk_models import RiskModel, VaRCalculator
from .performance import PerformanceAnalyzer
from .visualization import PortfolioVisualizer
from .analyzer import Analyzer

__all__ = [
    "Portfolio",
    "DataProvider", 
    "PortfolioOptimizer",
    "RiskModel",
    "VaRCalculator",
    "PerformanceAnalyzer",
    "PortfolioVisualizer",
    "Analyzer",
]
