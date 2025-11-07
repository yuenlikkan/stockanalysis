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
    for attempt in range(max_retries):
        try:
            # Create new Ticker instance each try (helps avoid stale sessions)
            intc = yf.Ticker(ticker)
            
            # Get price history
            hist = intc.history(start=start, end=end, interval="1d")
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
    
    return None, None

def main():
    # Calculate date range (3 years from today)
    end = dt.date.today()
    start = end - dt.timedelta(days=3*365)  # rough 3 years

    print(f"Fetching Intel (INTC) data from {start} to {end}...")
    
    # Fetch data with retries
    hist, info = fetch_with_retry("INTC", start, end)
    
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