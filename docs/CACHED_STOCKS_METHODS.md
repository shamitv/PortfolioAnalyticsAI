# Cached Stocks Methods - DataProvider Enhancement

## Overview
Added three new methods to the `DataProvider` class to query and understand what stock symbols are available in the local cache database.

## New Methods Added

### 1. `get_cached_stocks(include_etfs: bool = True) -> List[str]`
Returns a list of all stock symbols available in the cache.

**Parameters:**
- `include_etfs` (bool): If True, includes ETF symbols. If False, filters out ETFs.

**Returns:**
- List of stock symbols found in the cache database

**Example:**
```python
data_provider = DataProvider(cache=True)

# Get all symbols (including ETFs)
all_symbols = data_provider.get_cached_stocks(include_etfs=True)
print(f"Total symbols: {len(all_symbols)}")

# Get only stocks (excluding ETFs)
stocks_only = data_provider.get_cached_stocks(include_etfs=False)
print(f"Stock symbols: {len(stocks_only)}")
```

### 2. `get_cached_symbols_info() -> Dict[str, Dict[str, Union[str, int]]]`
Returns detailed information about all symbols available in cache.

**Returns:**
- Dictionary with symbol as key and info dict as value containing:
  - `'count'`: Number of data points available
  - `'start_date'`: Earliest date available  
  - `'end_date'`: Latest date available
  - `'symbol_type'`: 'ETF' or 'Stock' (classification)

**Example:**
```python
symbols_info = data_provider.get_cached_symbols_info()

for symbol, info in symbols_info.items():
    print(f"{symbol}: {info['symbol_type']}, "
          f"{info['count']} data points, "
          f"{info['start_date']} to {info['end_date']}")
```

### 3. Enhanced ETF Detection
The methods leverage the existing `get_cached_etfs()` method for accurate ETF classification using:
- Sector metadata table (authoritative ETF symbols)
- Common known ETF symbols
- ETF naming pattern recognition

## Implementation Details

### Database Structure
The methods query the existing `price_data` table:
```sql
CREATE TABLE price_data (
    Date TEXT,
    Symbol TEXT,
    Close REAL,
    PRIMARY KEY (Date, Symbol)
)
```

### Error Handling
- Graceful handling when cache is not enabled
- Proper exception handling for database errors
- Returns empty results instead of raising exceptions
- Debug logging for troubleshooting

### Performance Considerations
- Single SQL queries for efficiency
- Leverages existing ETF detection logic
- Minimal memory footprint for large datasets

## Use Cases

### 1. Portfolio Construction
```python
# Get available stocks for portfolio optimization
available_stocks = data_provider.get_cached_stocks(include_etfs=False)
selected_stocks = available_stocks[:20]  # Select first 20 stocks
```

### 2. Data Availability Check
```python
# Check what data is available before analysis
symbols_info = data_provider.get_cached_symbols_info()
well_covered_symbols = [
    symbol for symbol, info in symbols_info.items() 
    if info['count'] > 1000  # More than 1000 data points
]
```

### 3. Asset Classification
```python
# Separate stocks and ETFs for different analysis
all_symbols = data_provider.get_cached_stocks(include_etfs=True)
stocks_only = data_provider.get_cached_stocks(include_etfs=False)
etfs_only = data_provider.get_cached_etfs()

print(f"Total: {len(all_symbols)}, Stocks: {len(stocks_only)}, ETFs: {len(etfs_only)}")
```

### 4. Data Quality Analysis
```python
# Analyze data coverage and quality
symbols_info = data_provider.get_cached_symbols_info()

# Find symbols with most complete data
best_coverage = max(symbols_info.values(), key=lambda x: x['count'])
print(f"Best data coverage: {best_coverage['count']} data points")

# Find date range of available data
all_start_dates = [info['start_date'] for info in symbols_info.values()]
all_end_dates = [info['end_date'] for info in symbols_info.values()]
print(f"Data range: {min(all_start_dates)} to {max(all_end_dates)}")
```

## Testing
- Comprehensive test suite in `tests/test_cached_stocks.py`
- Tests cover normal operation, edge cases, and error conditions
- Mock database setup for isolated testing
- All tests passing

## Integration
These methods integrate seamlessly with existing DataProvider functionality:
- Respect existing cache settings and database connections
- Use the same debug logging system
- Follow established error handling patterns
- Compatible with existing ETF detection logic

## Benefits
1. **Discovery**: Easily discover what stock data is available
2. **Planning**: Plan analysis based on available data coverage
3. **Classification**: Distinguish between stocks and ETFs
4. **Quality**: Assess data quality and completeness
5. **Efficiency**: Query cache without loading actual price data

The new methods provide essential functionality for understanding and working with cached financial data, making the DataProvider more comprehensive and user-friendly.
