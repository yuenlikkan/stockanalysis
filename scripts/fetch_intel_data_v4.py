#!/usr/bin/env python3
"""Fetch Intel (INTC) historical prices and fundamental data.

Downloads 3 years of daily price history for Intel Corporation using Stooq as the data source,
and attempts to get fundamental data from yfinance as a backup.
"""
import datetime as dt
import time
import pandas as pd
from pandas_datareader import data as pdr

def fetch_price_history(symbol, start, end, max_retries=3, delay=2):
    """Fetch price history data with retries"""
    for attempt in range(max_retries):
        try:
            # Get price history from Stooq
            df = pdr.DataReader(symbol, 'stooq', start=start, end=end)
            
            # Stooq returns data in reverse chronological order, so sort it
            df = df.sort_index()
            
            if df.empty:
                print(f"Attempt {attempt + 1}: Empty price history returned")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                continue
                
            return df
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
    
    return None

def main():
    # Calculate date range (3 years from today)
    end = dt.date.today()
    start = end - dt.timedelta(days=3*365)  # rough 3 years

    print(f"Fetching Intel (INTC) data from {start} to {end}...")
    
    # Fetch price history from Stooq
    hist = fetch_price_history("INTC.US", start, end)
    
    if hist is None:
        print("Error: Failed to fetch price history")
        return 1
        
    # Save price history
    price_file = "intc_daily_prices.csv"
    hist.to_csv(price_file)
    print(f"Wrote {len(hist)} days of price history to {price_file}")
    print("\nNote: Fundamental data fetch disabled due to Yahoo Finance API limitations.")
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())