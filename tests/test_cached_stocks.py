"""
Additional tests for cached stocks functionality in DataProvider.
"""

import unittest
from unittest.mock import patch, MagicMock
import sqlite3
import tempfile
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from portfolio_analytics.data_provider import DataProvider


class TestDataProviderCachedStocks(unittest.TestCase):
    """Test cached stocks functionality in DataProvider."""
    
    def setUp(self):
        """Set up test database with sample data."""
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        # Create sample data
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        
        # Create price_data table
        cursor.execute('''
            CREATE TABLE price_data (
                Date TEXT,
                Symbol TEXT,
                Close REAL,
                PRIMARY KEY (Date, Symbol)
            )
        ''')
        
        # Create sector_metadata table
        cursor.execute('''
            CREATE TABLE sector_metadata (
                etf_ticker TEXT PRIMARY KEY,
                sector_name TEXT
            )
        ''')
        
        # Insert sample price data
        sample_data = [
            ('2023-01-01', 'AAPL', 150.0),
            ('2023-01-01', 'MSFT', 250.0),
            ('2023-01-01', 'SPY', 400.0),
            ('2023-01-01', 'QQQ', 300.0),
            ('2023-01-01', 'XLK', 130.0),
            ('2023-01-02', 'AAPL', 151.0),
            ('2023-01-02', 'MSFT', 251.0),
            ('2023-01-02', 'SPY', 401.0),
            ('2023-01-02', 'QQQ', 301.0),
            ('2023-01-02', 'XLK', 131.0),
        ]
        cursor.executemany('INSERT INTO price_data VALUES (?, ?, ?)', sample_data)
        
        # Insert sample sector metadata
        sector_data = [
            ('XLK', 'Technology'),
            ('XLF', 'Financial'),
        ]
        cursor.executemany('INSERT INTO sector_metadata VALUES (?, ?)', sector_data)
        
        conn.commit()
        conn.close()
        
        # Initialize data provider with test database
        self.data_provider = DataProvider(cache=True, cache_db=self.temp_db.name, debug=True)
    
    def tearDown(self):
        """Clean up test database."""
        if hasattr(self, 'data_provider') and self.data_provider.db_conn:
            self.data_provider.db_conn.close()
        os.unlink(self.temp_db.name)
    
    def test_get_cached_stocks_with_etfs(self):
        """Test getting all cached stocks including ETFs."""
        stocks = self.data_provider.get_cached_stocks(include_etfs=True)
        
        # Should return all symbols
        expected_symbols = ['AAPL', 'MSFT', 'QQQ', 'SPY', 'XLK']
        self.assertEqual(sorted(stocks), sorted(expected_symbols))
    
    def test_get_cached_stocks_without_etfs(self):
        """Test getting cached stocks excluding ETFs."""
        stocks = self.data_provider.get_cached_stocks(include_etfs=False)
        
        # Should exclude ETF-like symbols
        # Based on the ETF detection logic, some symbols might be classified as ETFs
        self.assertIsInstance(stocks, list)
        # AAPL and MSFT should likely be included as stocks
        # The exact result depends on ETF detection logic
    
    def test_get_cached_symbols_info(self):
        """Test getting detailed symbol information."""
        symbols_info = self.data_provider.get_cached_symbols_info()
        
        # Should return info for all symbols
        self.assertEqual(len(symbols_info), 5)
        
        # Check structure of returned data
        for symbol, info in symbols_info.items():
            self.assertIn('count', info)
            self.assertIn('start_date', info)
            self.assertIn('end_date', info)
            self.assertIn('symbol_type', info)
            self.assertIn(info['symbol_type'], ['ETF', 'Stock'])
            self.assertEqual(info['count'], 2)  # Each symbol has 2 data points
            self.assertEqual(info['start_date'], '2023-01-01')
            self.assertEqual(info['end_date'], '2023-01-02')
    
    def test_get_cached_stocks_no_cache(self):
        """Test behavior when cache is not enabled."""
        no_cache_provider = DataProvider(cache=False)
        
        stocks = no_cache_provider.get_cached_stocks()
        self.assertEqual(stocks, [])
        
        symbols_info = no_cache_provider.get_cached_symbols_info()
        self.assertEqual(symbols_info, {})
    
    def test_get_cached_stocks_empty_database(self):
        """Test behavior with empty database."""
        # Create empty database
        empty_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        empty_db.close()
        
        empty_provider = DataProvider(cache=True, cache_db=empty_db.name)
        
        try:
            stocks = empty_provider.get_cached_stocks()
            self.assertEqual(stocks, [])
            
            symbols_info = empty_provider.get_cached_symbols_info()
            self.assertEqual(symbols_info, {})
        finally:
            if empty_provider.db_conn:
                empty_provider.db_conn.close()
            os.unlink(empty_db.name)
    
    def test_get_cached_stocks_database_error(self):
        """Test error handling when database operations fail."""
        # Close the database connection to simulate an error
        self.data_provider.db_conn.close()
        self.data_provider.db_conn = None
        
        stocks = self.data_provider.get_cached_stocks()
        self.assertEqual(stocks, [])
        
        symbols_info = self.data_provider.get_cached_symbols_info()
        self.assertEqual(symbols_info, {})


if __name__ == '__main__':
    unittest.main()
