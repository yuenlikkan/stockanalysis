#!/usr/bin/env python3
"""Fetch Intel (INTC) historical prices and fundamental data.

Downloads 3 years of daily price history and fundamental data for Intel Corporation
and saves both to CSV files in the workspace root.
"""
import datetime as dt
import time
import pandas as pd
import yfinance as yf

def fetch_with_retry(ticker, start, end, max_retries=3, delay=2):
    """Fetch data with retries on failure"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for attempt in range(max_retries):
        try:
            # Create new Ticker instance each try
            intc = yf.Ticker(ticker)
            
            # Configure session
            intc.session.headers.update(headers)
            
            # Get price history
            hist = intc.history(period="3y", interval="1d")  # Use period instead of start/end
            if hist.empty:
                print(f"Attempt {attempt + 1}: Empty price history returned")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                continue
                
            # Get fundamental data
            info = intc.info
            if not info:
                print(f"Attempt {attempt + 1}: No fundamental data returned")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                continue
                
            return hist, info
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
    
    return None, None

def main():
    print("Fetching Intel (INTC) data for last 3 years...")
    
    # Fetch data with retries
    hist, info = fetch_with_retry("INTC", None, None)  # start/end not needed with period
    
    if hist is None or info is None:
        print("Error: Failed to fetch data after all retries")
        return 1
        
    fundamentals = {
        'name': info.get('longName', ''),
        'sector': info.get('sector', ''),
        'industry': info.get('industry', ''),
        'market_cap': info.get('marketCap', ''),
        'pe_ratio': info.get('trailingPE', ''),
        'forward_pe': info.get('forwardPE', ''),
        'peg_ratio': info.get('pegRatio', ''),
        'dividend_yield': info.get('dividendYield', ''),
        'beta': info.get('beta', ''),
        'book_value': info.get('bookValue', ''),
        'price_to_book': info.get('priceToBook', ''),
        'profit_margins': info.get('profitMargins', ''),
        'revenue_growth': info.get('revenueGrowth', ''),
        'revenue_per_share': info.get('revenuePerShare', ''),
        'ebitda': info.get('ebitda', ''),
        'debt_to_equity': info.get('debtToEquity', ''),
        'current_ratio': info.get('currentRatio', ''),
        'return_on_equity': info.get('returnOnEquity', ''),
        'quarterly_revenue_growth': info.get('quarterlyRevenueGrowth', ''),
        'quarterly_earnings_growth': info.get('quarterlyEarningsGrowth', ''),
    }

    # Save price history
    price_file = "intc_daily_prices.csv"
    hist.to_csv(price_file)
    print(f"Wrote {len(hist)} days of price history to {price_file}")
    
    # Save fundamentals 
    fund_file = "intc_fundamentals.csv"
    pd.DataFrame([fundamentals]).to_csv(fund_file, index=False)
    print(f"Wrote fundamental data to {fund_file}")
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())