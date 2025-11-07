#!/usr/bin/env python3
"""Test API access for various symbols to identify which ones are restricted."""

import requests
import time

# Finnhub API configuration
API_KEY = "d46kqa1r01qgc9etapv0d46kqa1r01qgc9etapvg"
BASE_URL = "https://finnhub.io/api/v1"

def test_symbol_access(symbol):
    """Test access to a specific symbol"""
    url = f"{BASE_URL}/quote"
    params = {
        "symbol": symbol,
        "token": API_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if 'c' in data and data['c'] is not None:
                return "✓ SUCCESS"
            else:
                return f"✗ NO DATA: {data}"
        else:
            return f"✗ HTTP {response.status_code}: {response.text}"
    except Exception as e:
        return f"✗ ERROR: {e}"

def main():
    # Test various symbols including some that work and some that might not
    test_symbols = [
        'AAPL',   # Should work
        'GOOGL',  # Should work
        'ADBE',   # The problematic one
        'MSFT',   # Should work
        'AMD',    # Should work
        'NVDA',   # Test another popular one
        'TSLA',   # Test another one
    ]
    
    print("Testing API access for various symbols:")
    print("=" * 50)
    
    for symbol in test_symbols:
        result = test_symbol_access(symbol)
        print(f"{symbol:6} | {result}")
        time.sleep(1.2)  # Rate limiting
    
    print("\nTesting candle data (historical prices) for ADBE:")
    # Test candle data specifically for ADBE
    url = f"{BASE_URL}/stock/candle"
    params = {
        "symbol": "ADBE",
        "resolution": "D",
        "from": 1699831200,  # Recent timestamp
        "to": 1730885000,    # More recent timestamp
        "token": API_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        print(f"Candle API Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Candle API Response: {data}")
        else:
            print(f"Candle API Error: {response.text}")
    except Exception as e:
        print(f"Candle API Exception: {e}")

if __name__ == "__main__":
    main()