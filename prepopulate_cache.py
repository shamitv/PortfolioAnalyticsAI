"""
Pre-populates the market data cache from S&P 500 company list.
"""
import pandas as pd
from portfolio_analytics.data_provider import DataProvider
from datetime import datetime, timedelta
import os
import sqlite3
import pandas_market_calendars as mcal


def create_company_list_table(conn):
    """Create the company_list table if it doesn't exist."""
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS company_list (
            Symbol TEXT PRIMARY KEY,
            Name TEXT,
            Sector TEXT
        )
    ''')
    conn.commit()

def create_sector_metadata_table(conn):
    """Create the sector_metadata table if it doesn't exist."""
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sector_metadata (
            sector_name TEXT PRIMARY KEY,
            etf_ticker TEXT,
            etf_name TEXT,
            description TEXT,
            created_date TEXT,
            updated_date TEXT
        )
    ''')
    conn.commit()



def get_us_trading_holidays():
    """
    Returns a list of US trading holidays from 2000 to current year.
    """    
    # Initialize NYSE calendar
    nyse = mcal.get_calendar('NYSE')
    
    # Get holidays from 2000 to current year
    current_year = datetime.now().year
    start_year = 2000
    
    # Get holidays for the date range
    start_date = f'{start_year}-01-01'
    end_date = f'{current_year}-12-31'
    holidays = nyse.holidays(start_date=start_date, end_date=end_date)
    
    # Return list of holiday dates from 2000 to current year
    return holidays.to_list()



def populate_sector_metadata(conn, sector_etfs_path):
    """Populate the sector_metadata table from the Sector_ETFs.csv file."""
    if not os.path.exists(sector_etfs_path):
        print("Sector_ETFs.csv not found, skipping sector metadata population.")
        return
    
    df = pd.read_csv(sector_etfs_path)
    current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Prepare data for insertion
    sector_data = []
    for _, row in df.iterrows():
        sector_data.append({
            'sector_name': row['Sector'],
            'etf_ticker': row['Ticker'],
            'etf_name': row['ETF Name'],
            'description': f"SPDR sector ETF tracking {row['Sector']} sector",
            'created_date': current_date,
            'updated_date': current_date
        })
    
    # Convert to DataFrame and insert into database
    sector_df = pd.DataFrame(sector_data)
    sector_df.to_sql('sector_metadata', conn, if_exists='replace', index=False)
    print(f"Populated sector metadata with {len(sector_df)} sectors.")

def populate_company_list(conn, excel_path):
    """Populate the company_list table from an Excel file."""
    df = pd.read_excel(excel_path, sheet_name='basics')
    # Ensure columns are named correctly for the database table
    df = df[['Symbol', 'Name', 'Sector']]
    df.to_sql('company_list', conn, if_exists='replace', index=False)
    print(f"Populated company list with {len(df)} companies.")

def prepopulate_market_data_cache():
    """
    Iterates over the S&P 500 list and downloads historical data
    to populate the cache.
    """
    sample_data_dir = 'sample_data'
    db_path = os.path.join(sample_data_dir, 'market_data.db')
    excel_path = os.path.join(sample_data_dir, 'snp_500_companies.xlsx')

    # Ensure the sample_data directory exists
    os.makedirs(sample_data_dir, exist_ok=True)

    # Copy sample data from the package to get current cache of market data
    temp_data_provider = DataProvider(debug=True)
    temp_data_provider.copy_sample_data(sample_data_dir)
    print("Copied current cache of market data from package sample data.")

    # Connect to the database and create the company list table
    conn = sqlite3.connect(db_path)
    create_company_list_table(conn)
    create_sector_metadata_table(conn)
    populate_company_list(conn, excel_path)
    
    # Populate sector metadata table
    sector_etfs_path = os.path.join(sample_data_dir, 'Sector_ETFs.csv')
    populate_sector_metadata(conn, sector_etfs_path)
    conn.close()

    # Initialize DataProvider with caching enabled and debug output on
    data_provider = DataProvider(cache=True, cache_db=db_path, debug=True)

    # Get the list of symbols from the database
    conn = sqlite3.connect(db_path)
    symbols_df = pd.read_sql_query("SELECT Symbol FROM company_list", conn)
    symbols = symbols_df['Symbol'].tolist()
    conn.close()

    # Also get sector ETF symbols from the CSV file
    sector_etfs_path = os.path.join(sample_data_dir, 'Sector_ETFs.csv')
    if os.path.exists(sector_etfs_path):
        sector_etfs_df = pd.read_csv(sector_etfs_path)
        sector_etf_symbols = sector_etfs_df['Ticker'].tolist()
        print(f"Found {len(sector_etf_symbols)} sector ETFs to download.")
    else:
        sector_etf_symbols = []
        print("Sector_ETFs.csv not found, skipping sector ETF data download.")

    # Get risk-free rate metadata and add the symbol
    risk_free_metadata = data_provider.get_risk_free_rate_metadata()
    risk_free_symbol = [risk_free_metadata['symbol']]
    print(f"Adding risk-free rate symbol: {risk_free_metadata['symbol']} ({risk_free_metadata['name']})")

    # Combine S&P 500 symbols, sector ETF symbols, and risk-free rate symbol
    all_symbols = symbols + sector_etf_symbols + risk_free_symbol
    print(f"Total symbols to fetch: {len(all_symbols)} ({len(symbols)} S&P 500 + {len(sector_etf_symbols)} sector ETFs + 1 risk-free rate)")

    # Define date range
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=18*365)

    print(f"Fetching data for {len(all_symbols)} symbols from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")

    # Iterate over all symbols (S&P 500 + sector ETFs + risk-free rate) and fetch data
    for i, symbol in enumerate(all_symbols):
        if symbol in sector_etf_symbols:
            symbol_type = "sector ETF"
        elif symbol in risk_free_symbol:
            symbol_type = "risk-free rate"
        else:
            symbol_type = "S&P 500 stock"
        
        try:
            print(f"({i+1}/{len(all_symbols)}) Fetching data for {symbol} ({symbol_type})...")
            data_provider.get_price_data(
                symbols=symbol,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
        except Exception as e:
            print(f"Could not fetch data for {symbol}: {e}")

    print("\nCache pre-population complete.")

    # Print the size of the database file
    db_size = os.path.getsize(db_path)
    print(f"Database file size: {db_size / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    prepopulate_market_data_cache()
