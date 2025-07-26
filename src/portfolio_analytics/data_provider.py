"""
Data provider for fetching financial data from various sources.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from typing import List, Optional, Dict, Union
from datetime import datetime, timedelta
import sqlite3
import os


class DataProvider:
    """
    A unified interface for fetching financial data from multiple sources.
    
    Currently supports Yahoo Finance with plans to add other data sources.
    """
    
    def __init__(self, source: str = "yahoo", cache: bool = False, cache_db: str = 'portfolio_cache.db'):
        """
        Initialize data provider.
        
        Args:
            source: Data source ('yahoo', 'alpha_vantage', 'quandl')
            cache: If True, cache data to a local SQLite database.
            cache_db: Path to the SQLite database file.
        """
        self.source = source
        self._validate_source()
        self.cache = cache
        self.db_conn = None
        if self.cache:
            self.db_conn = sqlite3.connect(cache_db)
            self._create_cache_table()

    def __del__(self):
        if self.db_conn:
            self.db_conn.close()

    def _create_cache_table(self):
        """Create the cache table if it doesn't exist."""
        if self.db_conn:
            cursor = self.db_conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS price_data (
                    Date TEXT,
                    Symbol TEXT,
                    Close REAL,
                    PRIMARY KEY (Date, Symbol)
                )
            ''')
            self.db_conn.commit()
    
    def _validate_source(self) -> None:
        """Validate the data source."""
        supported_sources = ["yahoo", "alpha_vantage", "quandl"]
        if self.source not in supported_sources:
            raise ValueError(f"Source '{self.source}' not supported. "
                           f"Supported sources: {supported_sources}")
    
    def get_price_data(self, 
                      symbols: Union[str, List[str]],
                      start_date: str = "2020-01-01",
                      end_date: Optional[str] = None,
                      interval: str = "1d") -> pd.DataFrame:
        """
        Fetch historical price data for given symbols.
        
        Args:
            symbols: Stock symbol(s) to fetch data for
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format), defaults to today
            interval: Data interval ('1d', '1wk', '1mo')
        
        Returns:
            DataFrame with historical prices (adjusted close)
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        if self.cache:
            try:
                cached_data = self._load_from_cache(symbols, start_date, end_date)
                if not cached_data.empty:
                    # Check if all symbols are present and data covers the full range
                    if all(s in cached_data.columns for s in symbols):
                         if cached_data.index.min().strftime('%Y-%m-%d') <= start_date and cached_data.index.max().strftime('%Y-%m-%d') >= end_date:
                            return cached_data[symbols]

            except Exception as e:
                print(f"Cache read failed: {e}")

        
        if self.source == "yahoo":
            data = self._fetch_yahoo_data(symbols, start_date, end_date, interval)
            if self.cache:
                self._save_to_cache(data)
            return data
        else:
            raise NotImplementedError(f"Data fetching for {self.source} not yet implemented")

    def _load_from_cache(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Load price data from cache."""
        if not self.db_conn:
            return pd.DataFrame()

        query = f"""
            SELECT Date, Symbol, Close FROM price_data
            WHERE Symbol IN ({','.join(['?']*len(symbols))})
            AND Date >= ? AND Date <= ?
        """
        params = symbols + [start_date, end_date]
        df = pd.read_sql_query(query, self.db_conn, params=params)
        
        if df.empty:
            return pd.DataFrame()

        df['Date'] = pd.to_datetime(df['Date'])
        pivot_df = df.pivot(index='Date', columns='Symbol', values='Close')
        return pivot_df

    def _save_to_cache(self, data: pd.DataFrame):
        """Save price data to cache."""
        if not self.db_conn or data.empty:
            return

        df_to_save = data.stack().reset_index()
        df_to_save.columns = ['Date', 'Symbol', 'Close']
        df_to_save['Date'] = df_to_save['Date'].dt.strftime('%Y-%m-%d')

        try:
            df_to_save.to_sql('price_data', self.db_conn, if_exists='append', index=False, method=self._upsert_sqlite)
        except Exception as e:
            print(f"Failed to cache data: {e}")

    def _upsert_sqlite(self, table, conn, keys, data_iter):
        from sqlalchemy.dialects.sqlite import insert
        from sqlalchemy import table as sql_table, column
        
        sql_table_obj = sql_table(table.name, column('Date'), column('Symbol'), column('Close'))
        
        for data in data_iter:
            stmt = insert(sql_table_obj).values(data)
            stmt = stmt.on_conflict_do_update(
                index_elements=['Date', 'Symbol'],
                set_={'Close': stmt.excluded.Close}
            )
            conn.execute(stmt)
    
    def _fetch_yahoo_data(self, 
                         symbols: List[str],
                         start_date: str,
                         end_date: str,
                         interval: str) -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date string
            end_date: End date string
            interval: Data interval
        
        Returns:
            DataFrame with adjusted close prices
        """
        try:
            # Download data for all symbols
            data = yf.download(
                tickers=symbols,
                start=start_date,
                end=end_date,
                interval=interval,
                group_by='ticker',
                auto_adjust=True,
                prepost=True,
                threads=True,
                proxy=None
            )
            
            if len(symbols) == 1:
                # Single symbol - return Close column
                if 'Close' in data.columns:
                    result = pd.DataFrame({symbols[0]: data['Close']})
                else:
                    result = pd.DataFrame({symbols[0]: data})
            else:
                # Multiple symbols - extract Close prices
                close_prices = {}
                for symbol in symbols:
                    if symbol in data.columns.get_level_values(0):
                        close_prices[symbol] = data[symbol]['Close']
                    else:
                        print(f"Warning: No data found for symbol {symbol}")
                
                result = pd.DataFrame(close_prices)
            
            # Remove any rows with all NaN values
            result = result.dropna(how='all')
            
            if result.empty:
                raise ValueError("No valid data retrieved for the given symbols and date range")
            
            return result
            
        except Exception as e:
            raise ValueError(f"Error fetching data from Yahoo Finance: {str(e)}")
    
    def get_company_info(self, symbol: str) -> Dict:
        """
        Get company information for a given symbol.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Dictionary with company information
        """
        if self.source == "yahoo":
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Extract key information
                company_info = {
                    'symbol': symbol,
                    'name': info.get('longName', 'N/A'),
                    'sector': info.get('sector', 'N/A'),
                    'industry': info.get('industry', 'N/A'),
                    'market_cap': info.get('marketCap', 'N/A'),
                    'pe_ratio': info.get('trailingPE', 'N/A'),
                    'dividend_yield': info.get('dividendYield', 'N/A'),
                    'beta': info.get('beta', 'N/A'),
                    'description': info.get('longBusinessSummary', 'N/A')
                }
                
                return company_info
                
            except Exception as e:
                raise ValueError(f"Error fetching company info for {symbol}: {str(e)}")
        else:
            raise NotImplementedError(f"Company info for {self.source} not yet implemented")
    
    def get_financial_ratios(self, symbol: str) -> Dict:
        """
        Get financial ratios for a given symbol.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Dictionary with financial ratios
        """
        if self.source == "yahoo":
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                ratios = {
                    'symbol': symbol,
                    'pe_ratio': info.get('trailingPE'),
                    'forward_pe': info.get('forwardPE'),
                    'pb_ratio': info.get('priceToBook'),
                    'ps_ratio': info.get('priceToSalesTrailing12Months'),
                    'debt_to_equity': info.get('debtToEquity'),
                    'current_ratio': info.get('currentRatio'),
                    'quick_ratio': info.get('quickRatio'),
                    'roe': info.get('returnOnEquity'),
                    'roa': info.get('returnOnAssets'),
                    'profit_margin': info.get('profitMargins'),
                    'operating_margin': info.get('operatingMargins'),
                    'gross_margin': info.get('grossMargins')
                }
                
                return ratios
                
            except Exception as e:
                raise ValueError(f"Error fetching financial ratios for {symbol}: {str(e)}")
        else:
            raise NotImplementedError(f"Financial ratios for {self.source} not yet implemented")
    
    def get_dividend_data(self, 
                         symbol: str,
                         start_date: str = "2020-01-01",
                         end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get dividend data for a given symbol.
        
        Args:
            symbol: Stock symbol
            start_date: Start date for dividend history
            end_date: End date for dividend history
        
        Returns:
            DataFrame with dividend data
        """
        if self.source == "yahoo":
            try:
                ticker = yf.Ticker(symbol)
                dividends = ticker.dividends
                
                if end_date is None:
                    end_date = datetime.now().strftime("%Y-%m-%d")
                
                # Filter by date range
                dividends = dividends[start_date:end_date]
                
                return pd.DataFrame({symbol: dividends})
                
            except Exception as e:
                raise ValueError(f"Error fetching dividend data for {symbol}: {str(e)}")
        else:
            raise NotImplementedError(f"Dividend data for {self.source} not yet implemented")
    
    def validate_symbols(self, symbols: List[str]) -> Dict[str, bool]:
        """
        Validate if symbols exist and have data available.
        
        Args:
            symbols: List of stock symbols to validate
        
        Returns:
            Dictionary mapping symbols to their validity status
        """
        validation_results = {}
        
        for symbol in symbols:
            try:
                # Try to fetch a small amount of recent data
                test_data = self.get_price_data(
                    symbols=symbol,
                    start_date=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
                )
                validation_results[symbol] = not test_data.empty
            except:
                validation_results[symbol] = False
        
        return validation_results
    
    def get_market_indices(self, 
                          indices: List[str] = ["^GSPC", "^DJI", "^IXIC"],
                          start_date: str = "2020-01-01",
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get market index data.
        
        Args:
            indices: List of index symbols (default: S&P 500, Dow Jones, NASDAQ)
            start_date: Start date
            end_date: End date
        
        Returns:
            DataFrame with index prices
        """
        return self.get_price_data(
            symbols=indices,
            start_date=start_date,
            end_date=end_date
        )
