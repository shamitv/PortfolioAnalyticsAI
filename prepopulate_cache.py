"""
Pre-populates the market data cache from S&P 500 company list.
"""
import pandas as pd
from portfolio_analytics.data_provider import DataProvider
from datetime import datetime, timedelta
import os
import sqlite3

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

def populate_company_list(conn, csv_path):
    """Populate the company_list table from a CSV file."""
    df = pd.read_csv(csv_path)
    df.to_sql('company_list', conn, if_exists='replace', index=False)
    print(f"Populated company list with {len(df)} companies.")

def prepopulate_market_data_cache():
    """
    Iterates over the S&P 500 list and downloads historical data
    to populate the cache.
    """
    sample_data_dir = 'sample_data'
    db_path = os.path.join(sample_data_dir, 'market_data.db')
    csv_path = os.path.join(sample_data_dir, 'snp_500_companies.csv')

    # Ensure the sample_data directory exists
    os.makedirs(sample_data_dir, exist_ok=True)

    # Connect to the database and create the company list table
    conn = sqlite3.connect(db_path)
    create_company_list_table(conn)
    populate_company_list(conn, csv_path)
    conn.close()

    # Initialize DataProvider with caching enabled and debug output on
    data_provider = DataProvider(cache=True, cache_db=db_path, debug=True)

    # Get the list of symbols from the database
    conn = sqlite3.connect(db_path)
    symbols_df = pd.read_sql_query("SELECT Symbol FROM company_list", conn)
    symbols = symbols_df['Symbol'].tolist()
    conn.close()

    # Define date range
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=18*365)

    print(f"Fetching data for {len(symbols)} symbols from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")

    # Iterate over symbols and fetch data
    for i, symbol in enumerate(symbols):
        try:
            print(f"({i+1}/{len(symbols)}) Fetching data for {symbol}...")
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
