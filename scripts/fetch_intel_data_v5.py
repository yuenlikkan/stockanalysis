#!/usr/bin/env python3
"""Fetch Intel (INTC) historical prices and fundamental data.

Downloads daily price history and fundamental data for Intel Corporation
using Alpha Vantage as the data source.
"""
import datetime as dt
import time
import json
import requests
import pandas as pd
import numpy as np

# Alpha Vantage API configuration
API_KEY = "NHEM32XSRU17MJFF"
BASE_URL = "https://www.alphavantage.co/query"

def fetch_daily_prices(symbol, max_retries=3, delay=2):
    """Fetch daily price history with retries"""
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": "full",  # Get full history
        "apikey": API_KEY
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(BASE_URL, params=params)
            data = response.json()
            
            if "Time Series (Daily)" not in data:
                print(f"Attempt {attempt + 1}: Invalid response - {data.get('Note', data.get('Error Message', 'Unknown error'))}")
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
            
            return df
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
    
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
                print(f"Attempt {attempt + 1}: Invalid response - {data.get('Note', data.get('Error Message', 'Unknown error'))}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                continue
                
            return data
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
    
    return None

def main():
    symbol = "INTC"
    print(f"Fetching Intel ({symbol}) data...")
    
    # Fetch daily price history
    print("Fetching price history...")
    hist = fetch_daily_prices(symbol)
    
    if hist is None:
        print("Error: Failed to fetch price history")
        return 1
        
    # Save price history
    price_file = "intc_daily_prices.csv"
    hist.to_csv(price_file)
    print(f"Wrote {len(hist)} days of price history to {price_file}")
    
    # Fetch fundamental data
    print("\nFetching fundamental data...")
    info = fetch_company_overview(symbol)
    
    if info:
        # Convert numeric strings to floats where possible
        for key, value in info.items():
            try:
                if value and isinstance(value, str) and not value.isalpha():
                    info[key] = float(value)
            except ValueError:
                pass
        
        # Save fundamentals
        fund_file = "intc_fundamentals.csv"
        pd.DataFrame([info]).to_csv(fund_file, index=False)
        print(f"Wrote fundamental data to {fund_file}")
    else:
        print("Warning: Could not fetch fundamental data")
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())