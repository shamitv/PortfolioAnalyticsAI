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



    def set_trading_holidays(self, holidays: List[str]) -> None:
        """
        Set the list of trading holidays.
        
        Args:
            holidays: List of holiday dates in 'YYYY-MM-DD' format.
        """
        if holidays:
            self.trading_holidays = [pd.to_datetime(holiday).date() for holiday in holidays]
        else:
            self.trading_holidays = []


    def copy_sample_data(self, target_dir: str) -> None:
        """
        Copy sample data files bundled with the package to a specified directory.
        
        Args:
            target_dir: Path to the directory where sample data will be copied.
        
        Raises:
            FileNotFoundError: If the sample data directory does not exist.
            OSError: If copying files fails.
        """
        import shutil
        sample_data_dir = os.path.join(os.path.dirname(__file__), 'sample_data')
        if not os.path.isdir(sample_data_dir):
            raise FileNotFoundError(f"Sample data directory not found: {sample_data_dir}")
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
        for filename in os.listdir(sample_data_dir):
            src_path = os.path.join(sample_data_dir, filename)
            dst_path = os.path.join(target_dir, filename)
            if os.path.isfile(src_path):
                try:
                    shutil.copy2(src_path, dst_path)
                    if self.debug:
                        print(f"Copied {src_path} to {dst_path}")
                except FileExistsError:
                    if self.debug:
                        print(f"File already exists: {dst_path}")
                except Exception as e:
                    raise OSError(f"Failed to copy {src_path} to {dst_path}: {e}")
    """
    A unified interface for fetching financial data from multiple sources.
    
    Currently supports Yahoo Finance with plans to add other data sources.
    """
    
    def __init__(self, source: str = "yahoo", cache: bool = False, cache_db: str = 'portfolio_cache.db', debug: bool = False, trading_holidays: Optional[List[str]] = None):
        """
        Initialize data provider.
        
        Args:
            source: Data source ('yahoo', 'alpha_vantage', 'quandl')
            cache: If True, cache data to a local SQLite database.
            cache_db: Path to the SQLite database file.
            debug: If True, print debug/troubleshooting messages.
            trading_holidays: List of trading holiday dates in 'YYYY-MM-DD' format.
        """
        self.source = source
        self._validate_source()
        self.cache = cache
        self.db_conn = None
        self.debug = debug
        self.trading_holidays = []
        
        # Set trading holidays if provided
        if trading_holidays:
            self.set_trading_holidays(trading_holidays)
        
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
        Fetch historical price data for given symbols, utilizing cache if enabled.
        
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

        all_symbol_data = []

        for symbol in symbols:
            symbol_data_df = pd.DataFrame()
            if self.cache:
                # Attempt to load whatever exists in cache for the symbol
                cached_data = self._load_from_cache([symbol], start_date, end_date)
                if not cached_data.empty:
                    symbol_data_df = cached_data

            # Determine what data is missing
            missing_ranges = self._get_missing_date_ranges(symbol_data_df, start_date, end_date)

            if missing_ranges:
                if self.debug:
                    print(f"Fetching {symbol} for missing ranges: {missing_ranges} from {self.source}.")
                fetched_data_list = []
                for r_start, r_end in missing_ranges:
                    if self.source == "yahoo":
                        fetched_range_data = self._fetch_yahoo_data([symbol], r_start, r_end, interval)
                        if not fetched_range_data.empty:
                            fetched_data_list.append(fetched_range_data)
                    else:
                        raise NotImplementedError(f"Data fetching for {self.source} not yet implemented")

                if fetched_data_list:
                    newly_fetched_data = pd.concat(fetched_data_list)
                    if self.cache:
                        self._save_to_cache(newly_fetched_data)
                    
                    # Combine cached and newly fetched data
                    symbol_data_df = pd.concat([symbol_data_df, newly_fetched_data]).sort_index()
                    # Remove duplicates, keeping the newly fetched data
                    symbol_data_df = symbol_data_df[~symbol_data_df.index.duplicated(keep='last')]

            else:
                if self.debug:
                    print(f"Loading {symbol} from cache.")

            if not symbol_data_df.empty:
                all_symbol_data.append(symbol_data_df)

        if not all_symbol_data:
            return pd.DataFrame()

        # Combine all dataframes
        final_df = pd.concat(all_symbol_data, axis=1)
        return final_df.sort_index()

    def _get_missing_date_ranges(self, df: pd.DataFrame, start_date: str, end_date: str) -> List[tuple]:
        """
        Identifies missing date ranges in a dataframe, considering trading holidays and weekends.
        Ignores gaps in the middle of existing data to avoid fetching data for temporary data issues.
        """
        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)
        
        if df.empty:
            return [(start_date, end_date)]

        # Get business days for the full range (excludes weekends)
        full_range = pd.bdate_range(start=start_date_dt, end=end_date_dt)
        
        # Remove trading holidays if they are set
        if self.trading_holidays:
            # Convert date objects back to datetime for comparison
            trading_holidays_dt = [pd.to_datetime(holiday) for holiday in self.trading_holidays]
            # Only consider holidays that fall on business days (Monday-Friday) and within our range
            business_day_holidays = []
            for holiday in trading_holidays_dt:
                if holiday.weekday() < 5 and start_date_dt <= holiday <= end_date_dt:  # Monday=0, Friday=4
                    business_day_holidays.append(holiday)
            
            # Remove holidays from the full range
            if business_day_holidays:
                full_range = full_range.difference(pd.DatetimeIndex(business_day_holidays))
        
        # Find dates that are missing from the dataframe's index
        missing_dates = full_range.difference(df.index)

        if missing_dates.empty:
            return []

        # Get the actual data range (first and last dates with data)
        if not df.empty:
            data_start = df.index.min()
            data_end = df.index.max()
            
            # Only fetch data for ranges that extend beyond existing data
            # Ignore gaps in the middle of the dataset
            ranges_to_fetch = []
            
            # Check if we need data before the first available data point
            missing_before = missing_dates[missing_dates < data_start]
            if not missing_before.empty:
                ranges_to_fetch.append((missing_before.min().strftime('%Y-%m-%d'), 
                                      missing_before.max().strftime('%Y-%m-%d')))
            
            # Check if we need data after the last available data point
            missing_after = missing_dates[missing_dates > data_end]
            if not missing_after.empty:
                ranges_to_fetch.append((missing_after.min().strftime('%Y-%m-%d'), 
                                      missing_after.max().strftime('%Y-%m-%d')))
            
            if self.debug and len(missing_dates) > len(missing_before) + len(missing_after):
                gap_count = len(missing_dates) - len(missing_before) - len(missing_after)
                print(f"Ignoring {gap_count} missing days in the middle of existing data (gaps)")
            
            return ranges_to_fetch
        else:
            # No existing data, group consecutive missing dates into ranges
            gaps = []
            if not missing_dates.empty:
                # Find blocks of consecutive dates
                breaks = np.where(np.diff(missing_dates.to_julian_date()) > 1)[0] + 1
                # Split the array of missing dates at these breaks
                date_blocks = np.split(missing_dates, breaks)
                
                for block in date_blocks:
                    if not block.empty:
                        gaps.append((block[0].strftime('%Y-%m-%d'), block[-1].strftime('%Y-%m-%d')))
            
            return gaps

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
        # conn is a pandas SQLDatabase object, which has an execute method
        for data in data_iter:
            # Assuming data is a tuple in the order of columns
            # The columns are Date, Symbol, Close
            conn.execute(f'''
                INSERT INTO {table.name} (Date, Symbol, Close)
                VALUES (?, ?, ?)
                ON CONFLICT(Date, Symbol) DO UPDATE SET Close=excluded.Close
            ''', data)
    
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
        # yfinance end date is exclusive, so we need to add one day to it.
        end_date_dt = pd.to_datetime(end_date) + timedelta(days=1)
        end_date_str = end_date_dt.strftime('%Y-%m-%d')

        try:
            # yfinance sometimes raises YFPricesMissingError for single days that are holidays
            # We can suppress this and just return an empty dataframe.
            data = yf.download(
                tickers=symbols,
                start=start_date,
                end=end_date_str,
                interval=interval,
                group_by='ticker',
                auto_adjust=True,
                prepost=True,
                threads=True,
                proxy=None
            )
        except Exception as e:
            # This is a broad exception, but yfinance can be unpredictable
            # For our purpose of filling cache gaps, failing silently is acceptable.
            if self.debug:
                print(f"Could not download data for {symbols} from {start_date} to {end_date}: {e}")
            return pd.DataFrame(columns=symbols)
            
        if data.empty:
            return pd.DataFrame(columns=symbols)

        # The 'group_by' argument creates a multi-level column index.
        # We need to extract the 'Close' price for each symbol.
        close_prices = {}
        for symbol in symbols:
            # Check if the symbol exists in the downloaded data
            if symbol in data.columns:
                # For single symbol downloads, the structure is simpler
                if isinstance(data.columns, pd.MultiIndex):
                    symbol_close = data[symbol]['Close']
                else:
                    symbol_close = data['Close']
                
                # Remove NaN values which represent days with no trading
                close_prices[symbol] = symbol_close.dropna()

        if not close_prices:
            return pd.DataFrame(columns=symbols)

        # Combine the 'Close' price series into a single DataFrame
        result_df = pd.DataFrame(close_prices)
        result_df.index.name = 'Date'
        return result_df

    def get_company_info(self, symbol: str) -> Optional[Dict]:
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
    
    def get_risk_free_rate_metadata(self) -> Dict[str, str]:
        """
        Return metadata about the risk-free return instrument.
        
        Returns:
            Dictionary containing metadata about the risk-free rate proxy:
            - symbol: The ticker symbol
            - name: Official name of the instrument
            - description: Detailed description of the instrument and its use
            - data_source: Source of the data
            - currency: Currency denomination
            - frequency: Data frequency
        """
        return {
            'symbol': '^IRX',
            'name': '13-week (3-month) U.S. Treasury bill yield',
            'description': (
                'CBOE 13-Week Treasury Bill Yield Index, which reflects the annualized '
                'market yield on U.S. three-month Treasury bills and is widely used as a '
                'proxy for the short-term risk-free rate in financial models. The index '
                'is officially named "13 WEEK TREASURY BILL (^IRX)" and is provided by '
                'CBOE Global Indices, tracking the yield on three-month U.S. Treasury bills '
                'quoted in USD. These bills are sold at a discount to par in weekly auctions '
                'and their yield represents the smallest return investors accept for virtually '
                'risk-free, short-term lending to the U.S. government.'
            ),
            'data_source': 'CBOE Global Indices via Yahoo Finance',
            'currency': 'USD',
            'frequency': 'Daily',
            'asset_type': 'Treasury Bill Yield',
            'maturity': '3 months (13 weeks)',
            'issuer': 'U.S. Treasury Department'
        }
