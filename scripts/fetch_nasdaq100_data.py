#!/usr/bin/env python3
"""Fetch historical prices and fundamental data for Nasdaq-100 stocks.

Downloads daily price history and fundamental data for all stocks in nasdaq100_cached.csv
using Alpha Vantage as the data source. Creates a database for later analysis.
"""
import datetime as dt
import time
import json
import os
from pathlib import Path
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm

# Alpha Vantage API configuration
API_KEY = "NHEM32XSRU17MJFF"
BASE_URL = "https://www.alphavantage.co/query"

# API rate limiting (5 calls per minute for free tier)
CALLS_PER_MINUTE = 5
SECONDS_BETWEEN_CALLS = 60 / CALLS_PER_MINUTE

def fetch_daily_prices(symbol, max_retries=3, delay=2):
    """Fetch daily price history with retries"""
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": "full",
        "apikey": API_KEY
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(BASE_URL, params=params)
            data = response.json()
            
            if "Time Series (Daily)" not in data:
                error_msg = data.get('Note', data.get('Error Message', 'Unknown error'))
                print(f"\nAttempt {attempt + 1} for {symbol}: Invalid response - {error_msg}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                continue
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
            
            # Rename columns
            df.columns = [c.split(". ")[1].lower() for c in df.columns]
            
            # Convert strings to floats
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
                
            # Sort by date ascending
            df = df.sort_index()
            df.index = pd.to_datetime(df.index)
            
            return df
            
        except Exception as e:
            print(f"\nAttempt {attempt + 1} for {symbol} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
    
    return None

def fetch_company_overview(symbol, max_retries=3, delay=2):
    """Fetch company overview/fundamental data with retries"""
    params = {
        "function": "OVERVIEW",
        "symbol": symbol,
        "apikey": API_KEY
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(BASE_URL, params=params)
            data = response.json()
            
            if not data or "Symbol" not in data:
                error_msg = data.get('Note', data.get('Error Message', 'Unknown error'))
                print(f"\nAttempt {attempt + 1} for {symbol}: Invalid response - {error_msg}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                continue
                
            # Convert numeric strings to floats where possible
            for key, value in data.items():
                try:
                    if value and isinstance(value, str) and not value.isalpha():
                        data[key] = float(value)
                except ValueError:
                    pass
                    
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

def main():
    # Create data directories
    data_dir = "stock_data"
    prices_dir = os.path.join(data_dir, "daily_prices")
    fundamentals_dir = os.path.join(data_dir, "fundamentals")
    ensure_dir(prices_dir)
    ensure_dir(fundamentals_dir)
    
    # Read Nasdaq-100 tickers
    try:
        tickers = pd.read_csv("nasdaq100_cached.csv", header=None)[0].tolist()
    except Exception as e:
        print(f"Error reading nasdaq100_cached.csv: {e}")
        return 1
    
    print(f"Found {len(tickers)} tickers in Nasdaq-100 list")
    
    # Track successful downloads
    price_success = []
    fundamental_success = []
    
    # Process each ticker
    for ticker in tqdm(tickers, desc="Processing tickers"):
        # Rate limiting
        time.sleep(SECONDS_BETWEEN_CALLS)
        
        # Fetch price history
        hist = fetch_daily_prices(ticker)
        if hist is not None:
            price_file = os.path.join(prices_dir, f"{ticker}_daily.csv")
            hist.to_csv(price_file)
            price_success.append(ticker)
            
        # Rate limiting
        time.sleep(SECONDS_BETWEEN_CALLS)
        
        # Fetch fundamental data
        info = fetch_company_overview(ticker)
        if info is not None:
            fund_file = os.path.join(fundamentals_dir, f"{ticker}_fundamentals.csv")
            pd.DataFrame([info]).to_csv(fund_file, index=False)
            fundamental_success.append(ticker)
    
    # Create summary report
    print("\nDownload Summary:")
    print(f"Price history downloaded: {len(price_success)}/{len(tickers)}")
    print(f"Fundamental data downloaded: {len(fundamental_success)}/{len(tickers)}")
    
    if len(price_success) < len(tickers) or len(fundamental_success) < len(tickers):
        failed_prices = set(tickers) - set(price_success)
        failed_fundamentals = set(tickers) - set(fundamental_success)
        
        if failed_prices:
            print("\nFailed price downloads:")
            print(", ".join(sorted(failed_prices)))
            
        if failed_fundamentals:
            print("\nFailed fundamental downloads:")
            print(", ".join(sorted(failed_fundamentals)))
    
    # Create combined CSVs for analysis
    print("\nCreating combined datasets...")
    
    # Combine fundamental data
    all_fundamentals = []
    for ticker in fundamental_success:
        try:
            fund_file = os.path.join(fundamentals_dir, f"{ticker}_fundamentals.csv")
            df = pd.read_csv(fund_file)
            all_fundamentals.append(df)
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