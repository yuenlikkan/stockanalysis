#!/usr/bin/env python3
"""Analyze Nasdaq-100 stocks: Price vs 200-day Moving Average.

Creates a comprehensive database showing:
- Last stock price
- 200-day moving average
- Percentage of last price relative to 200-day MA
- Additional technical indicators

Reads price data from stock_data/daily_prices/ directory.
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def calculate_moving_average(prices, window=200):
    """Calculate moving average for given window"""
    return prices.rolling(window=window, min_periods=window).mean()

def calculate_technical_indicators(df):
    """Calculate various technical indicators"""
    # Ensure we have enough data for 200-day MA
    if len(df) < 200:
        return None
    
    # Get the close prices
    close_prices = df['close']
    
    # Calculate 200-day moving average
    ma_200 = calculate_moving_average(close_prices, 200)
    
    # Get the most recent values
    last_price = close_prices.iloc[-1]
    last_ma_200 = ma_200.iloc[-1]
    
    # Calculate percentage relative to 200-day MA
    price_to_ma_pct = ((last_price / last_ma_200) - 1) * 100 if not pd.isna(last_ma_200) else np.nan
    
    # Additional useful indicators
    ma_50 = calculate_moving_average(close_prices, 50)
    last_ma_50 = ma_50.iloc[-1]
    
    # Calculate volatility (20-day)
    volatility = close_prices.rolling(20).std().iloc[-1]
    
    # Calculate daily return
    daily_return = ((last_price / close_prices.iloc[-2]) - 1) * 100 if len(close_prices) > 1 else 0
    
    # Volume information
    avg_volume = df['volume'].rolling(20).mean().iloc[-1]
    last_volume = df['volume'].iloc[-1]
    
    # Price range (52-week high/low)
    if len(df) >= 252:  # ~1 year of trading days
        year_data = df.tail(252)
    else:
        year_data = df
    
    week_52_high = year_data['high'].max()
    week_52_low = year_data['low'].min()
    
    return {
        'last_price': round(last_price, 2),
        'ma_200': round(last_ma_200, 2) if not pd.isna(last_ma_200) else None,
        'ma_50': round(last_ma_50, 2) if not pd.isna(last_ma_50) else None,
        'price_to_ma200_pct': round(price_to_ma_pct, 2) if not pd.isna(price_to_ma_pct) else None,
        'daily_return_pct': round(daily_return, 2),
        'volatility_20d': round(volatility, 2) if not pd.isna(volatility) else None,
        'volume': int(last_volume) if not pd.isna(last_volume) else None,
        'avg_volume_20d': int(avg_volume) if not pd.isna(avg_volume) else None,
        '52w_high': round(week_52_high, 2),
        '52w_low': round(week_52_low, 2),
        'data_points': len(df),
        'last_date': df.index[-1].strftime('%Y-%m-%d')
    }

def analyze_single_stock(ticker, data_dir):
    """Analyze a single stock and return its metrics"""
    csv_file = os.path.join(data_dir, f"{ticker}_daily.csv")
    
    if not os.path.exists(csv_file):
        print(f"Warning: No data file found for {ticker}")
        return None
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        
        # Ensure we have the required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: Missing required columns for {ticker}")
            return None
        
        # Sort by date to ensure chronological order
        df = df.sort_index()
        
        # Calculate indicators
        indicators = calculate_technical_indicators(df)
        
        if indicators is None:
            print(f"Warning: Insufficient data for {ticker} (need at least 200 days)")
            return None
        
        # Add ticker symbol
        indicators['ticker'] = ticker
        
        return indicators
        
    except Exception as e:
        print(f"Error processing {ticker}: {str(e)}")
        return None

def create_analysis_database():
    """Create the main analysis database"""
    
    # Define data directory
    data_dir = os.path.join("stock_data", "daily_prices")
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        print("Please run fetch_nasdaq100_prices_yfinance.py first to download data.")
        return None
    
    # Get list of tickers from nasdaq100_cached.csv or from files
    try:
        tickers = pd.read_csv("nasdaq100_cached.csv", header=None)[0].tolist()
        print(f"Found {len(tickers)} tickers in nasdaq100_cached.csv")
    except:
        # Fallback: get tickers from existing files
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('_daily.csv')]
        tickers = [f.replace('_daily.csv', '') for f in csv_files]
        print(f"Found {len(tickers)} tickers from existing data files")
    
    if not tickers:
        print("Error: No tickers found")
        return None
    
    # Analyze each stock
    results = []
    failed_tickers = []
    
    print(f"\nAnalyzing {len(tickers)} stocks...")
    for i, ticker in enumerate(tickers, 1):
        print(f"Processing {ticker} ({i}/{len(tickers)})", end='\r')
        
        analysis = analyze_single_stock(ticker, data_dir)
        if analysis:
            results.append(analysis)
        else:
            failed_tickers.append(ticker)
    
    print(f"\nCompleted analysis of {len(results)} stocks")
    
    if failed_tickers:
        print(f"Failed to analyze: {', '.join(failed_tickers)}")
    
    # Create DataFrame from results
    if not results:
        print("Error: No successful analyses")
        return None
    
    df = pd.DataFrame(results)
    
    # Reorder columns for better readability
    column_order = [
        'ticker', 'last_price', 'ma_200', 'price_to_ma200_pct', 
        'ma_50', 'daily_return_pct', 'volatility_20d',
        'volume', 'avg_volume_20d', '52w_high', '52w_low',
        'data_points', 'last_date'
    ]
    
    df = df[column_order]
    
    # Sort by price_to_ma200_pct descending (stocks most above their 200-day MA first)
    df = df.sort_values('price_to_ma200_pct', ascending=False, na_position='last')
    
    return df

def print_summary_statistics(df):
    """Print summary statistics about the analysis"""
    print(f"\n{'='*80}")
    print("NASDAQ-100 ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    total_stocks = len(df)
    above_ma200 = len(df[df['price_to_ma200_pct'] > 0])
    below_ma200 = len(df[df['price_to_ma200_pct'] < 0])
    
    print(f"Total stocks analyzed: {total_stocks}")
    print(f"Stocks above 200-day MA: {above_ma200} ({above_ma200/total_stocks*100:.1f}%)")
    print(f"Stocks below 200-day MA: {below_ma200} ({below_ma200/total_stocks*100:.1f}%)")
    
    # Top and bottom performers
    print(f"\nTop 5 performers vs 200-day MA:")
    top_5 = df.head(5)[['ticker', 'last_price', 'ma_200', 'price_to_ma200_pct']]
    for _, row in top_5.iterrows():
        print(f"  {row['ticker']:5} | ${row['last_price']:8.2f} | MA200: ${row['ma_200']:8.2f} | {row['price_to_ma200_pct']:+6.1f}%")
    
    print(f"\nBottom 5 performers vs 200-day MA:")
    bottom_5 = df.tail(5)[['ticker', 'last_price', 'ma_200', 'price_to_ma200_pct']]
    for _, row in bottom_5.iterrows():
        print(f"  {row['ticker']:5} | ${row['last_price']:8.2f} | MA200: ${row['ma_200']:8.2f} | {row['price_to_ma200_pct']:+6.1f}%")
    
    # Statistics
    if 'price_to_ma200_pct' in df.columns:
        avg_pct = df['price_to_ma200_pct'].mean()
        median_pct = df['price_to_ma200_pct'].median()
        print(f"\nAverage price vs MA200: {avg_pct:+.1f}%")
        print(f"Median price vs MA200: {median_pct:+.1f}%")

def main():
    """Main function"""
    
    # Check for command line arguments
    show_all = '--all' in sys.argv
    export_csv = '--csv' in sys.argv
    
    print("Nasdaq-100 Stock Analysis: Price vs 200-day Moving Average")
    print("="*60)
    
    # Create the analysis database
    df = create_analysis_database()
    
    if df is None:
        return 1
    
    # Print summary statistics
    print_summary_statistics(df)
    
    # Display results
    if show_all:
        print(f"\n{'='*80}")
        print("COMPLETE ANALYSIS RESULTS")
        print(f"{'='*80}")
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_columns', None)
        print(df.to_string(index=False))
    else:
        print(f"\n{'='*80}")
        print("KEY METRICS (Top 20 stocks)")
        print(f"{'='*80}")
        print("Ticker | Last Price | 200-day MA | vs MA200 | Daily Return")
        print("-" * 60)
        for _, row in df.head(20).iterrows():
            print(f"{row['ticker']:6} | ${row['last_price']:9.2f} | ${row['ma_200']:9.2f} | {row['price_to_ma200_pct']:+7.1f}% | {row['daily_return_pct']:+7.1f}%")
        
        print(f"\nUse --all to see complete results")
    
    # Export to CSV if requested
    if export_csv:
        output_file = "nasdaq100_analysis.csv"
        df.to_csv(output_file, index=False)
        print(f"\nResults exported to {output_file}")
    
    print(f"\nAnalysis complete! Use --csv to export data to CSV file")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())