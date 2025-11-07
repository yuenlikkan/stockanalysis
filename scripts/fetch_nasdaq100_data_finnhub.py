#!/usr/bin/env python3
"""Fetch historical prices for Nasdaq-100 stocks using Finnhub.

Downloads 1 year of daily price history for all stocks in nasdaq100_cached.csv.
Always overwrites existing price data files to ensure fresh data.
Preserves existing fundamental data files (only fetches if missing).
Creates a structured database for analysis.

Usage:
    python fetch_nasdaq100_data_finnhub.py                    # Process all tickers (price data only)
    python fetch_nasdaq100_data_finnhub.py --ticker=SYMBOL    # Process single ticker
    python fetch_nasdaq100_data_finnhub.py --debug            # Show detailed API responses
"""
import datetime as dt
import time
import json
import os
import sys
from pathlib import Path
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm

# Finnhub API configuration
API_KEY = "d46kqa1r01qgc9etapv0d46kqa1r01qgc9etapvg"
BASE_URL = "https://finnhub.io/api/v1"

# API rate limiting (60 calls per minute for free tier)
CALLS_PER_MINUTE = 50  # Using 50 to be safe
SECONDS_BETWEEN_CALLS = 60 / CALLS_PER_MINUTE

# Debug mode
DEBUG = '--debug' in sys.argv

def get_unix_timestamp(date):
    """Convert datetime to Unix timestamp"""
    return int(time.mktime(date.timetuple()))

def fetch_daily_prices(symbol, max_retries=3, delay=2):
    """Fetch daily price history with retries"""
    # Calculate date range (1 year from today)
    end = dt.datetime.now()
    start = end - dt.timedelta(days=365)
    
    # Convert to Unix timestamps
    end_ts = get_unix_timestamp(end)
    start_ts = get_unix_timestamp(start)
    
    url = f"{BASE_URL}/stock/candle"
    params = {
        "symbol": symbol,
        "resolution": "D",  # Daily candles
        "from": start_ts,
        "to": end_ts,
        "token": API_KEY
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params)
            
            # Check HTTP status code first
            if response.status_code != 200:
                if response.status_code == 403:
                    print(f"\nSkipping {symbol}: Access denied (likely restricted in free tier)")
                    return "ACCESS_DENIED"  # Special return value for 403 errors
                print(f"\nAttempt {attempt + 1} for {symbol}: HTTP {response.status_code} - {response.text}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                continue
            
            data = response.json()
            
            if DEBUG:
                print(f"\nDEBUG: API Response for {symbol}: {data}")
            
            # Check for API error messages
            if 'error' in data:
                print(f"\nAttempt {attempt + 1} for {symbol}: API Error - {data['error']}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                continue
            
            if data.get('s') == 'no_data':
                print(f"\nAttempt {attempt + 1} for {symbol}: No data available")
                return None
                
            if 'c' not in data or not data['c']:  # 'c' is close prices
                print(f"\nAttempt {attempt + 1} for {symbol}: Invalid response - {data}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                continue
            
            # Create DataFrame
            df = pd.DataFrame({
                'open': data['o'],
                'high': data['h'],
                'low': data['l'],
                'close': data['c'],
                'volume': data['v']
            }, index=pd.to_datetime([dt.datetime.fromtimestamp(t) for t in data['t']]))
            
            # Sort by date ascending
            df = df.sort_index()
            
            return df
            
        except Exception as e:
            print(f"\nAttempt {attempt + 1} for {symbol} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
    
    return None

def fetch_company_info(symbol, max_retries=3, delay=2):
    """Fetch company information with retries"""
    url = f"{BASE_URL}/stock/profile2"
    params = {
        "symbol": symbol,
        "token": API_KEY
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params)
            
            # Check HTTP status code first
            if response.status_code != 200:
                if response.status_code == 403:
                    print(f"\nSkipping {symbol}: Access denied (likely restricted in free tier)")
                    return "ACCESS_DENIED"  # Special return value for 403 errors
                print(f"\nAttempt {attempt + 1} for {symbol}: HTTP {response.status_code} - {response.text}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                continue
            
            data = response.json()
            
            # Check for API error messages
            if 'error' in data:
                print(f"\nAttempt {attempt + 1} for {symbol}: API Error - {data['error']}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                continue
            
            if not data:
                print(f"\nAttempt {attempt + 1} for {symbol}: No data available")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                continue
            
            # Add basic financials
            metrics_url = f"{BASE_URL}/stock/metric"
            metrics_params = {
                "symbol": symbol,
                "metric": "all",
                "token": API_KEY
            }
            
            metrics_response = requests.get(metrics_url, params=metrics_params)
            metrics_data = metrics_response.json()
            
            if metrics_data and 'metric' in metrics_data:
                data.update(metrics_data['metric'])
                
            return data
            
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

def test_api_key():
    """Test if the API key is valid"""
    url = f"{BASE_URL}/quote"
    params = {
        "symbol": "AAPL",  # Test with a common stock
        "token": API_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if 'c' in data:  # Current price
                print("API key is valid and working")
                return True
            else:
                print("API key test failed - no data returned")
                return False
        else:
            print(f"API key test failed - HTTP {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"API key test failed - Error: {e}")
        return False

def main():
    # Check for specific ticker
    test_ticker = None
    for arg in sys.argv:
        if arg.startswith('--ticker='):
            test_ticker = arg.split('=')[1].upper()
    
    if test_ticker:
        print(f"Testing single ticker: {test_ticker}")
    
    # Test API key first
    print("Testing API key...")
    if not test_api_key():
        print("API key validation failed. Please check your API key.")
        return 1
    
    # Create data directories
    data_dir = "stock_data"
    prices_dir = os.path.join(data_dir, "daily_prices")
    fundamentals_dir = os.path.join(data_dir, "fundamentals")
    ensure_dir(prices_dir)
    ensure_dir(fundamentals_dir)
    
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
    
    # Track successful downloads and access restrictions
    price_success = []
    fundamental_success = []
    restricted_symbols = []
    
    # Process each ticker
    for ticker in tqdm(tickers, desc="Processing tickers"):
        # Always fetch and overwrite price data
        price_file = os.path.join(prices_dir, f"{ticker}_daily.csv")
        
        # Rate limiting
        time.sleep(SECONDS_BETWEEN_CALLS)
        
        # Fetch price history
        hist = fetch_daily_prices(ticker)
        if hist == "ACCESS_DENIED":
            restricted_symbols.append(ticker)
        elif hist is not None:
            hist.to_csv(price_file)
            price_success.append(ticker)
        
        # Skip fundamental data - assume existing files are good
        fund_file = os.path.join(fundamentals_dir, f"{ticker}_fundamentals.json")
        if os.path.exists(fund_file):
            fundamental_success.append(ticker)
        else:
            print(f"\nWarning: No existing fundamental data for {ticker}")
            # Only fetch fundamental data if no file exists
            time.sleep(SECONDS_BETWEEN_CALLS)
            info = fetch_company_info(ticker)
            if info == "ACCESS_DENIED":
                if ticker not in restricted_symbols:
                    restricted_symbols.append(ticker)
            elif info is not None:
                with open(fund_file, 'w') as f:
                    json.dump(info, f, indent=2)
                fundamental_success.append(ticker)
    
    # Create summary report
    print("\nDownload Summary:")
    print(f"Price history downloaded: {len(price_success)}/{len(tickers)}")
    print(f"Fundamental data downloaded: {len(fundamental_success)}/{len(tickers)}")
    
    if restricted_symbols:
        print(f"Restricted symbols (access denied): {len(restricted_symbols)}")
        print(f"Restricted: {', '.join(sorted(set(restricted_symbols)))}")
    
    if len(price_success) < len(tickers) or len(fundamental_success) < len(tickers):
        failed_prices = set(tickers) - set(price_success) - set(restricted_symbols)
        failed_fundamentals = set(tickers) - set(fundamental_success) - set(restricted_symbols)
        
        if failed_prices:
            print("\nFailed price downloads (other errors):")
            print(", ".join(sorted(failed_prices)))
            
        if failed_fundamentals:
            print("\nFailed fundamental downloads (other errors):")
            print(", ".join(sorted(failed_fundamentals)))
    
    # Create combined fundamental dataset
    print("\nCreating combined dataset...")
    all_fundamentals = []
    
    for ticker in fundamental_success:
        try:
            fund_file = os.path.join(fundamentals_dir, f"{ticker}_fundamentals.json")
            with open(fund_file, 'r') as f:
                data = json.load(f)
                # Flatten nested dictionaries for CSV format
                flat_data = {}
                for k, v in data.items():
                    if isinstance(v, (dict, list)):
                        flat_data[k] = json.dumps(v)
                    else:
                        flat_data[k] = v
                all_fundamentals.append(pd.DataFrame([flat_data]))
        except Exception as e:
            print(f"Error reading {ticker} fundamentals: {e}")
    
    if all_fundamentals:
        combined_fundamentals = pd.concat(all_fundamentals, ignore_index=True)
        combined_fundamentals.to_csv(os.path.join(data_dir, "all_fundamentals.csv"), index=False)
        print(f"Created combined fundamentals dataset with {len(combined_fundamentals)} companies")
    
    print("\nDone! Data is ready for analysis.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())