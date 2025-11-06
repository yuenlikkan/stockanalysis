# NASDAQ below 200-day MA filter

This small utility fetches NASDAQ-listed tickers and outputs those whose latest close price is below their 200-day moving average.

Prerequisites
- Python 3.8+
- PowerShell (Windows) or any shell

Install dependencies (PowerShell):

```powershell
python -m pip install -r requirements.txt
```

Run the script (PowerShell):

```powershell
python .\scripts\filter_below_200ma.py --out below_200ma.csv
```

Options
- `--out`: output CSV path (default `below_200ma.csv`)
- `--start`: history start date (YYYY-MM-DD). By default the script downloads ~400 days of history.
- `--chunk-size`: number of tickers to download per batch (default 100)
- `--threads`: threads parameter for yfinance (default 1)

Notes
- The script uses the NasdaqTrader file `nasdaqlisted.txt` to obtain tickers.
- yfinance may occasionally fail for some tickers; the script retries and skips problematic entries.
- This is intended as a simple screening tool; please backtest and validate before trading.
