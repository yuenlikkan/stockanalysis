#!/usr/bin/env python3
"""Fetch historical prices for Nasdaq-100 stocks using yfinance.

Downloads 1 year of daily price history for all stocks in nasdaq100_cached.csv.
Always overwrites existing price data files to ensure fresh data.
Uses yfinance which provides reliable free access to Yahoo Finance data.

Usage:
    python fetch_nasdaq100_prices_yfinance.py                    # Process all tickers
    python fetch_nasdaq100_prices_yfinance.py --ticker=SYMBOL    # Process single ticker
    python fetch_nasdaq100_prices_yfinance.py --debug            # Show detailed output
"""
import datetime as dt
import time
import os
import sys
from pathlib import Path
import pandas as pd
import yfinance as yf
from tqdm import tqdm

# Debug mode
DEBUG = '--debug' in sys.argv

def fetch_daily_prices(symbol, period="1y", max_retries=3, delay=1):
    """Fetch daily price history using yfinance with retries"""
    
    for attempt in range(max_retries):
        try:
            # Create yfinance ticker object
            ticker = yf.Ticker(symbol)
            
            # Fetch historical data
            hist = ticker.history(period=period, auto_adjust=True, back_adjust=True)
            
            if DEBUG:
                print(f"\nDEBUG: yfinance data for {symbol}: {len(hist)} rows")
            
            if hist.empty:
                print(f"\nAttempt {attempt + 1} for {symbol}: No data available")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                continue
            
            # Rename columns to match expected format
            hist.columns = [col.lower() for col in hist.columns]
            
            # Ensure we have the basic OHLCV columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in hist.columns for col in required_cols):
                print(f"\nAttempt {attempt + 1} for {symbol}: Missing required columns")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                continue
            
            # Sort by date ascending
            hist = hist.sort_index()
            
            if DEBUG:
                print(f"SUCCESS: {symbol} - {len(hist)} days of data from {hist.index[0].date()} to {hist.index[-1].date()}")
            
            return hist
            
        except Exception as e:
            print(f"\nAttempt {attempt + 1} for {symbol} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
    
    return None

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def main():
    # Check for specific ticker
    test_ticker = None
    for arg in sys.argv:
        if arg.startswith('--ticker='):
            test_ticker = arg.split('=')[1].upper()
    
    if test_ticker:
        print(f"Testing single ticker: {test_ticker}")
    
    # Create data directories
    data_dir = "stock_data"
    prices_dir = os.path.join(data_dir, "daily_prices")
    ensure_dir(prices_dir)
    
    # Read Nasdaq-100 tickers or use specific ticker
    if test_ticker:
        tickers = [test_ticker]
    else:
        try:
            tickers = pd.read_csv("nasdaq100_cached.csv", header=None)[0].tolist()
        except Exception as e:
            print(f"Error reading nasdaq100_cached.csv: {e}")
            return 1
    
    print(f"Found {len(tickers)} tickers in Nasdaq-100 list")
    
    # Track successful downloads
    price_success = []
    failed_tickers = []
    
    # Process each ticker
    for ticker in tqdm(tickers, desc="Processing tickers"):
        # Always fetch and overwrite price data
        price_file = os.path.join(prices_dir, f"{ticker}_daily.csv")
        
        # Small delay between requests to be respectful
        time.sleep(0.1)
        
        # Fetch price history
        hist = fetch_daily_prices(ticker)
        if hist is not None:
            hist.to_csv(price_file)
            price_success.append(ticker)
        else:
            failed_tickers.append(ticker)
    
    # Create summary report
    print("\nDownload Summary:")
    print(f"Price history downloaded: {len(price_success)}/{len(tickers)}")
    print(f"Success rate: {len(price_success)/len(tickers)*100:.1f}%")
    
    if failed_tickers:
        print(f"\nFailed downloads ({len(failed_tickers)}):")
        print(", ".join(sorted(failed_tickers)))
    
    print("\nDone! Price data is ready for analysis.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())