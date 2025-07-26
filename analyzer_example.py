"""
Example usage of the Analyzer class for generating comprehensive portfolio analysis
optimized for Vision-Capable LLMs.
"""

import pandas as pd
import numpy as np
from portfolio_analytics import Portfolio, DataProvider, Analyzer

def main():
    # Example portfolio setup
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # Portfolio weights
    
    # Create portfolio
    portfolio = Portfolio(symbols=symbols, weights=weights, name="Tech Portfolio")
    
    # Load data (you would use real data provider)
    data_provider = DataProvider()
    
    try:
        # Load portfolio data
        portfolio.load_data(data_provider, start_date="2022-01-01", end_date="2024-01-01")
        
        # Create analyzer
        analyzer = Analyzer(portfolio)
        
        # Generate comprehensive analysis for LLM
        print("Generating comprehensive analysis...")
        analysis_results = analyzer.export_for_llm(output_format="comprehensive")
        
        # Display key results
        print("\n" + "="*60)
        print("PORTFOLIO ANALYSIS SUMMARY")
        print("="*60)
        
        # Portfolio summary
        summary = analysis_results['portfolio_summary']
        print(f"\nPortfolio: {summary['portfolio_name']}")
        print(f"Assets: {', '.join(summary['assets'])}")
        print(f"Analysis Period: {summary['data_start_date']} to {summary['data_end_date']}")
        print(f"Total Observations: {summary['total_observations']}")
        
        # Key performance metrics
        perf_metrics = analysis_results['metrics']['performance']
        print(f"\nPERFORMANCE METRICS:")
        print(f"Annual Return: {perf_metrics.get('annual_return', 0):.2%}")
        print(f"Annual Volatility: {perf_metrics.get('annual_volatility', 0):.2%}")
        print(f"Sharpe Ratio: {perf_metrics.get('sharpe_ratio', 0):.3f}")
        print(f"Maximum Drawdown: {perf_metrics.get('max_drawdown', 0):.2%}")
        
        # Risk metrics
        risk_metrics = analysis_results['metrics']['risk']
        print(f"\nRISK METRICS:")
        print(f"VaR (95%): {risk_metrics.get('var_95', 0):.2%}")
        print(f"Expected Shortfall (95%): {risk_metrics.get('expected_shortfall_95', 0):.2%}")
        print(f"Skewness: {risk_metrics.get('skewness', 0):.3f}")
        print(f"Kurtosis: {risk_metrics.get('kurtosis', 0):.3f}")
        
        # Portfolio composition
        allocation = analysis_results['metrics']['allocation']
        print(f"\nPORTFOLIO COMPOSITION:")
        for symbol, weight in allocation['weights'].items():
            print(f"  {symbol}: {weight:.1%}")
        
        print(f"\nConcentration (HHI): {allocation['weight_concentration_hhi']:.3f}")
        print(f"Largest Position: {allocation['largest_position']:.1%}")
        print(f"Effective Number of Assets: {analysis_results['metrics']['portfolio']['effective_number_of_assets']:.1f}")
        
        # Greeks summary
        greeks = analysis_results['greeks']['portfolio_greeks']
        print(f"\nPORTFOLIO GREEKS:")
        print(f"Delta: {greeks.get('portfolio_delta', 0):.3f}")
        print(f"Gamma: {greeks.get('portfolio_gamma', 0):.3f}")
        print(f"Theta: {greeks.get('portfolio_theta', 0):.3f}")
        print(f"Vega: {greeks.get('portfolio_vega', 0):.3f}")
        print(f"Rho: {greeks.get('portfolio_rho', 0):.3f}")
        
        # Visualizations info
        viz_count = len(analysis_results['visualizations'])
        print(f"\nVISUALIZATIONS GENERATED: {viz_count}")
        print("Available charts:", ", ".join(analysis_results['visualizations'].keys()))
        
        print(f"\n" + "="*60)
        print("ANALYSIS COMPLETE - READY FOR LLM PROCESSING")
        print("="*60)
        
        # Example: Export specific format for different use cases
        print("\nExporting different formats...")
        
        # Summary format (key metrics + visuals)
        summary_export = analyzer.export_for_llm(output_format="summary")
        print(f"Summary format: {len(summary_export)} sections")
        
        # Metrics only (no visualizations)
        metrics_export = analyzer.export_for_llm(output_format="metrics_only")
        print(f"Metrics only: {len(metrics_export)} metric categories")
        
        # Visuals only (for image analysis)
        visuals_export = analyzer.export_for_llm(output_format="visuals_only")
        print(f"Visuals only: {len(visuals_export)} charts as base64 images")
        
        print(f"\nAnalysis timestamp: {summary['analysis_timestamp']}")
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        print("Note: This example requires actual market data to run completely.")
        print("The Analyzer class is ready to use with real portfolio data.")

def demonstrate_analyzer_features():
    """Demonstrate key features of the Analyzer class."""
    
    print("\nANALYZER CLASS FEATURES:")
    print("="*40)
    
    features = [
        "📊 Comprehensive Metrics Generation",
        "   • Performance metrics (returns, Sharpe, Sortino, etc.)",
        "   • Risk metrics (VaR, Expected Shortfall, drawdowns)",
        "   • Portfolio-specific metrics (concentration, diversification)",
        "   • Time-based analysis (monthly, quarterly, yearly)",
        "",
        "🔢 Portfolio Greeks Calculation", 
        "   • Delta (market sensitivity)",
        "   • Gamma (convexity)",
        "   • Theta (time decay)",
        "   • Vega (volatility sensitivity)",
        "   • Rho (interest rate sensitivity)",
        "",
        "📈 Visualization Generation",
        "   • Price history charts",
        "   • Returns distribution analysis",
        "   • Correlation matrix heatmaps",
        "   • Portfolio composition charts",
        "   • Cumulative returns comparison",
        "   • Drawdown analysis",
        "   • Risk-return scatter plots",
        "   • Rolling metrics charts",
        "   • Performance heatmaps",
        "   • Greek sensitivity analysis",
        "",
        "🤖 LLM-Optimized Output",
        "   • Base64 encoded images for vision models",
        "   • Structured JSON format",
        "   • Multiple export formats",
        "   • Comprehensive metadata",
        "",
        "📋 Export Formats",
        "   • 'comprehensive' - Full analysis with all components",
        "   • 'summary' - Key metrics + visualizations",
        "   • 'metrics_only' - Just the numbers and calculations",
        "   • 'visuals_only' - Just the charts as base64 images"
    ]
    
    for feature in features:
        print(feature)

if __name__ == "__main__":
    print("Portfolio Analytics AI - Analyzer Class Demo")
    print("="*50)
    
    demonstrate_analyzer_features()
    
    print("\n" + "="*50)
    print("RUNNING EXAMPLE (requires real data):")
    print("="*50)
    
    main()
