"""
Unit tests for DataProvider functionality including ETF detection and risk-free rate data.
"""

import unittest
import os
import sys
import tempfile
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from portfolio_analytics.data_provider import DataProvider


class TestDataProviderETFAndRiskFree(unittest.TestCase):
    """Test DataProvider ETF detection and risk-free rate functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db_path = self.temp_db.name
        self.temp_db.close()
        
        # Initialize test database with sample data
        self._setup_test_database()
        
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_db_path):
            os.unlink(self.temp_db_path)
    
    def _setup_test_database(self):
        """Create test database with sample data."""
        conn = sqlite3.connect(self.temp_db_path)
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
                sector_name TEXT PRIMARY KEY,
                etf_ticker TEXT,
                etf_name TEXT,
                description TEXT,
                created_date TEXT,
                updated_date TEXT
            )
        ''')
        
        # Insert sample sector ETF metadata
        sector_etfs = [
            ('Technology', 'XLK', 'Technology Select Sector SPDR Fund'),
            ('Financials', 'XLF', 'Financial Select Sector SPDR Fund'),
            ('Energy', 'XLE', 'Energy Select Sector SPDR Fund'),
            ('Health Care', 'XLV', 'Health Care Select Sector SPDR Fund'),
            ('Consumer Discretionary', 'XLY', 'Consumer Discretionary Select Sector SPDR Fund')
        ]
        
        current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for sector, ticker, name in sector_etfs:
            cursor.execute('''
                INSERT INTO sector_metadata 
                (sector_name, etf_ticker, etf_name, description, created_date, updated_date)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (sector, ticker, name, f"SPDR sector ETF tracking {sector} sector", current_date, current_date))
        
        # Insert sample price data for ETFs and other symbols
        test_symbols = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'SPY', 'QQQ', 'AAPL', 'GOOGL', '^IRX']
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        
        for symbol in test_symbols:
            for i, date in enumerate(dates):
                # Simple price generation for testing
                if symbol == '^IRX':  # Risk-free rate (yield data)
                    price = 4.5 + (i * 0.01)  # Yield around 4.5%
                else:
                    price = 100 + (i * 2) + hash(symbol) % 50  # Varied prices
                
                cursor.execute('''
                    INSERT INTO price_data (Date, Symbol, Close)
                    VALUES (?, ?, ?)
                ''', (date.strftime('%Y-%m-%d'), symbol, price))
        
        conn.commit()
        conn.close()
    
    def test_get_cached_etfs_with_sector_metadata(self):
        """Test getting ETF list from cache with sector metadata table."""
        # Initialize DataProvider with test database
        data_provider = DataProvider(cache=True, cache_db=self.temp_db_path, debug=True)
        
        # Get cached ETFs
        etf_list = data_provider.get_cached_etfs()
        
        # Verify we get the expected sector ETFs
        expected_sector_etfs = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY']
        self.assertTrue(len(etf_list) >= len(expected_sector_etfs))
        
        for etf in expected_sector_etfs:
            self.assertIn(etf, etf_list, f"Expected sector ETF {etf} not found in cached ETFs")
        
        # Note: The current implementation only returns sector ETFs from metadata table
        # This is actually correct behavior - we're testing the sector_metadata integration
        print(f"Found ETFs: {etf_list}")
        
        # Verify the ETFs are sorted
        self.assertEqual(etf_list, sorted(etf_list), "ETF list should be sorted")
    
    def test_get_cached_etfs_without_cache(self):
        """Test getting ETF list when cache is disabled."""
        # Initialize DataProvider without cache
        data_provider = DataProvider(cache=False, debug=True)
        
        # Get cached ETFs should return empty list
        etf_list = data_provider.get_cached_etfs()
        
        self.assertEqual(etf_list, [], "Should return empty list when cache is disabled")
    
    def test_get_cached_etfs_exception_handling(self):
        """Test ETF list retrieval with database errors."""
        # Initialize DataProvider with cache but handle connection error gracefully
        try:
            data_provider = DataProvider(cache=True, cache_db='/nonexistent/path/test.db', debug=True)
            # This should fail during initialization due to invalid path
            self.fail("Should have raised an exception for invalid database path")
        except sqlite3.OperationalError:
            # Expected behavior - invalid path should raise OperationalError
            pass
        
        # Test the case where we have a valid DataProvider but the get_cached_etfs method fails
        data_provider = DataProvider(cache=False, debug=True)  # No cache initially
        data_provider.cache = True  # Enable cache
        data_provider.db_conn = None  # But no connection
        
        # Should handle exception gracefully
        etf_list = data_provider.get_cached_etfs()
        
        self.assertEqual(etf_list, [], "Should return empty list on database error")
    
    def test_risk_free_rate_metadata(self):
        """Test getting risk-free rate metadata."""
        data_provider = DataProvider()
        
        metadata = data_provider.get_risk_free_rate_metadata()
        
        # Verify required fields
        required_fields = ['symbol', 'name', 'description', 'data_source', 'currency', 'frequency']
        for field in required_fields:
            self.assertIn(field, metadata, f"Missing required field: {field}")
        
        # Verify specific values
        self.assertEqual(metadata['symbol'], '^IRX', "Risk-free rate symbol should be ^IRX")
        self.assertEqual(metadata['currency'], 'USD', "Risk-free rate should be in USD")
        self.assertEqual(metadata['frequency'], 'Daily', "Risk-free rate should be daily frequency")
        self.assertIn('Treasury', metadata['name'], "Should reference Treasury in name")
        # Check for "three-month" instead of "3-month" as the description uses spelled-out form
        self.assertIn('three-month', metadata['description'], "Should mention three-month duration")
    
    @patch('portfolio_analytics.data_provider.yf.download')
    def test_get_risk_free_rate_data(self, mock_download):
        """Test fetching risk-free rate data."""
        # Setup mock data for risk-free rate
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        
        # Create mock data that mimics yfinance MultiIndex structure when group_by='ticker'
        # For single symbol, yfinance still creates MultiIndex with symbol as top level
        symbol = '^IRX'
        
        # Create MultiIndex columns: (symbol, price_type)
        columns = pd.MultiIndex.from_tuples([
            (symbol, 'Close'),
            (symbol, 'Open'), 
            (symbol, 'High'),
            (symbol, 'Low'),
            (symbol, 'Volume')
        ])
        
        mock_data = pd.DataFrame({
            (symbol, 'Close'): [4.5, 4.51, 4.52, 4.53, 4.54, 4.55, 4.56, 4.57, 4.58, 4.59],
            (symbol, 'Open'): [4.48, 4.50, 4.51, 4.52, 4.53, 4.54, 4.55, 4.56, 4.57, 4.58],
            (symbol, 'High'): [4.52, 4.53, 4.54, 4.55, 4.56, 4.57, 4.58, 4.59, 4.60, 4.61],
            (symbol, 'Low'): [4.47, 4.49, 4.50, 4.51, 4.52, 4.53, 4.54, 4.55, 4.56, 4.57],
            (symbol, 'Volume'): [0] * 10  # Treasury yields don't have volume
        }, index=dates, columns=columns)
        mock_data.index.name = 'Date'
        
        mock_download.return_value = mock_data
        
        # Initialize DataProvider
        data_provider = DataProvider(debug=True)
        
        # Get risk-free rate metadata
        risk_free_metadata = data_provider.get_risk_free_rate_metadata()
        risk_free_symbol = risk_free_metadata['symbol']
        
        # Fetch risk-free rate data
        risk_free_data = data_provider.get_price_data(
            symbols=risk_free_symbol,
            start_date='2024-01-01',
            end_date='2024-01-10'
        )
        
        # Verify data
        self.assertFalse(risk_free_data.empty, "Risk-free rate data should not be empty")
        self.assertIn(risk_free_symbol, risk_free_data.columns, f"Should contain {risk_free_symbol} column")
        self.assertEqual(len(risk_free_data), 10, "Should have 10 days of data")
        
        # Verify values are reasonable (yields typically 0-10%)
        values = risk_free_data[risk_free_symbol].dropna()
        self.assertTrue(all(0 <= v <= 10 for v in values), "Risk-free rate values should be reasonable yields")
    
    def test_get_risk_free_rate_from_cache(self):
        """Test getting risk-free rate data from cache."""
        # Initialize DataProvider with test database containing ^IRX data
        data_provider = DataProvider(cache=True, cache_db=self.temp_db_path, debug=True)
        
        # Get risk-free rate data from cache
        risk_free_data = data_provider.get_price_data(
            symbols='^IRX',
            start_date='2024-01-01',
            end_date='2024-01-10'
        )
        
        # Verify data
        self.assertFalse(risk_free_data.empty, "Risk-free rate data should be available from cache")
        self.assertIn('^IRX', risk_free_data.columns, "Should contain ^IRX column")
        
        # Verify we have the expected data range
        self.assertTrue(len(risk_free_data) > 0, "Should have risk-free rate data")
        
        # Values should be around 4.5% based on our test data
        values = risk_free_data['^IRX'].dropna()
        self.assertTrue(all(4.0 <= v <= 5.0 for v in values), "Risk-free rate values should be around 4.5%")
    
    def test_sector_etf_integration(self):
        """Test integration between sector metadata and ETF detection."""
        # Initialize DataProvider with test database
        data_provider = DataProvider(cache=True, cache_db=self.temp_db_path, debug=True)
        
        # Get cached ETFs
        etf_list = data_provider.get_cached_etfs()
        
        # Get sector ETFs from metadata
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT etf_ticker FROM sector_metadata")
        sector_etfs_from_db = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        # All sector ETFs should be in the cached ETF list
        for sector_etf in sector_etfs_from_db:
            self.assertIn(sector_etf, etf_list, f"Sector ETF {sector_etf} should be in cached ETF list")
        
        # Test that we can get price data for sector ETFs
        for sector_etf in sector_etfs_from_db[:2]:  # Test first 2 to save time
            price_data = data_provider.get_price_data(
                symbols=sector_etf,
                start_date='2024-01-01',
                end_date='2024-01-05'
            )
            self.assertFalse(price_data.empty, f"Should have price data for sector ETF {sector_etf}")
            self.assertIn(sector_etf, price_data.columns, f"Price data should contain {sector_etf} column")
    
    def test_data_provider_initialization_with_market_data_db(self):
        """Test DataProvider initialization with automatic market_data.db detection."""
        # Create a mock sample_data directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_data_dir = os.path.join(temp_dir, 'sample_data')
            os.makedirs(sample_data_dir)
            
            # Create a mock market_data.db file
            market_db_path = os.path.join(sample_data_dir, 'market_data.db')
            
            # Copy our test database to simulate market_data.db
            import shutil
            shutil.copy2(self.temp_db_path, market_db_path)
            
            # Mock the __file__ path to point to our temp directory
            with patch('portfolio_analytics.data_provider.os.path.dirname') as mock_dirname:
                mock_dirname.return_value = temp_dir
                
                # Initialize DataProvider without specifying cache_db
                data_provider = DataProvider(cache=True, debug=True)
                
                # Should automatically detect and use market_data.db
                etf_list = data_provider.get_cached_etfs()
                self.assertTrue(len(etf_list) > 0, "Should find ETFs from automatically detected market_data.db")


if __name__ == '__main__':
    unittest.main()
