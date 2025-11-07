#!/usr/bin/env python3
"""Fetch Intel (INTC) historical prices and fundamental data.

This script downloads 3 years of daily price history and current fundamental data
for Intel Corporation (INTC) using yfinance. Saves two CSV files:
- intel_history.csv: OHLCV daily data
- intel_info.csv: Current fundamental metrics (PE, market cap, etc.)
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

def fetch_intel_data(ticker="INTC"):
    """Fetch Intel price history and fundamentals using yfinance."""
    # Create Ticker object
    stock = yf.Ticker(ticker)
    
    # Calculate date range (3 years from today)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365)
    
    try:
        # Fetch daily price history
        hist_df = stock.history(start=start_date, end=end_date, interval="1d")
        if hist_df.empty:
            print(f"Error: No historical data received for {ticker}")
            return None, None
            
        # Get fundamental data via info()
        info = stock.info
        if not info:
            print(f"Error: No fundamental data received for {ticker}")
            return hist_df, None
            
        # Create a DataFrame with selected fundamental metrics
        metrics = [
            'symbol', 'longName', 'sector', 'industry',
            'marketCap', 'trailingPE', 'forwardPE', 'priceToBook',
            'profitMargins', 'operatingMargins', 'returnOnEquity',
            'totalRevenue', 'revenueGrowth', 'grossMargins',
            'ebitda', 'debtToEquity', 'dividendYield'
        ]
        
        # Extract available metrics (some may be missing)
        info_data = {metric: info.get(metric, None) for metric in metrics}
        info_df = pd.DataFrame([info_data])
        
        return hist_df, info_df
        
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return None, None

def main():
    """Main function: fetch data and save to CSVs."""
    print("Fetching Intel (INTC) data...")
    
    # Fetch both history and fundamentals
    hist_df, info_df = fetch_intel_data()
    
    if hist_df is None:
        print("Failed to fetch price history.")
        return 1
        
    # Save historical data
    hist_path = os.path.abspath("intel_history.csv")
    hist_df.to_csv(hist_path)
    print(f"Wrote {len(hist_df)} days of price history to {hist_path}")
    
    # Save fundamental data if available
    if info_df is not None:
        info_path = os.path.abspath("intel_info.csv")
        info_df.to_csv(info_path, index=False)
        print(f"Wrote fundamental data to {info_path}")
    else:
        print("Note: No fundamental data was retrieved.")
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())