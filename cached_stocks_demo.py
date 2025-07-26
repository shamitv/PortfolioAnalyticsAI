"""
Test script to demonstrate the new cached stocks methods in DataProvider.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from portfolio_analytics.data_provider import DataProvider

def main():
    """Demonstrate the new cached stocks methods."""
    
    print("DataProvider Cached Stocks Methods Demo")
    print("=" * 50)
    
    # Initialize data provider with cache enabled
    print("1. Initializing DataProvider with cache enabled...")
    data_provider = DataProvider(cache=True, debug=True)
    
    print("\n2. Getting all cached stocks (including ETFs)...")
    all_stocks = data_provider.get_cached_stocks(include_etfs=True)
    print(f"   Found {len(all_stocks)} total symbols")
    if all_stocks:
        print(f"   First 10 symbols: {all_stocks[:10]}")
        if len(all_stocks) > 10:
            print(f"   Last 5 symbols: {all_stocks[-5:]}")
    
    print("\n3. Getting cached stocks (excluding ETFs)...")
    stocks_only = data_provider.get_cached_stocks(include_etfs=False)
    print(f"   Found {len(stocks_only)} stock symbols (excluding ETFs)")
    if stocks_only:
        print(f"   First 10 stocks: {stocks_only[:10]}")
    
    print("\n4. Getting cached ETFs for comparison...")
    etfs_only = data_provider.get_cached_etfs()
    print(f"   Found {len(etfs_only)} ETF symbols")
    if etfs_only:
        print(f"   ETFs: {etfs_only}")
    
    print("\n5. Getting detailed symbol information...")
    symbols_info = data_provider.get_cached_symbols_info()
    print(f"   Retrieved detailed info for {len(symbols_info)} symbols")
    
    if symbols_info:
        print("\n   Sample symbol details:")
        sample_symbols = list(symbols_info.keys())[:5]
        for symbol in sample_symbols:
            info = symbols_info[symbol]
            print(f"   {symbol}: {info['symbol_type']}, "
                  f"{info['count']} data points, "
                  f"{info['start_date']} to {info['end_date']}")
    
    print("\n6. Summary statistics:")
    if symbols_info:
        total_symbols = len(symbols_info)
        etf_count = sum(1 for info in symbols_info.values() if info['symbol_type'] == 'ETF')
        stock_count = total_symbols - etf_count
        
        print(f"   Total symbols in cache: {total_symbols}")
        print(f"   ETFs: {etf_count}")
        print(f"   Stocks: {stock_count}")
        
        # Find symbol with most data points
        if symbols_info:
            max_data_symbol = max(symbols_info.keys(), key=lambda k: symbols_info[k]['count'])
            max_data_count = symbols_info[max_data_symbol]['count']
            print(f"   Symbol with most data: {max_data_symbol} ({max_data_count} data points)")
            
            # Find date range
            all_start_dates = [info['start_date'] for info in symbols_info.values()]
            all_end_dates = [info['end_date'] for info in symbols_info.values()]
            earliest_date = min(all_start_dates)
            latest_date = max(all_end_dates)
            print(f"   Data date range: {earliest_date} to {latest_date}")
    
    print("\nDemo completed! The DataProvider now provides comprehensive")
    print("methods to query and understand what stock data is available in cache.")

if __name__ == "__main__":
    main()
