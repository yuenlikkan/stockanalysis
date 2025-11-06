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
# Option A: use forward slashes (works cross-platform)
python ./scripts/filter_below_200ma.py --out below_200ma.csv

# Option B: PowerShell Windows path — escape backslashes or use a raw string in Python examples
python .\\scripts\\filter_below_200ma.py --out below_200ma.csv
```

Options
- `--out`: output CSV path (default `below_200ma.csv`)
- `--start`: history start date (YYYY-MM-DD). By default the script downloads ~400 days of history.
- `--chunk-size`: number of tickers to download per batch (default 100)
- `--threads`: threads parameter for yfinance (default 1)

- The script uses the NasdaqTrader file `nasdaqlisted.txt` to obtain tickers.
- yfinance may occasionally fail for some tickers; the script retries and skips problematic entries.
- This is intended as a simple screening tool; please backtest and validate before trading.
Notes
- The script uses the NasdaqTrader file `nasdaqlisted.txt` to obtain tickers.
- yfinance may occasionally fail for some tickers; the script retries and skips problematic entries.
- If you previously ran the script while the docstring contained a backslash escape like `"\s"`, Python may have emitted a SyntaxWarning that can persist in older .pyc files. If you still see the warning after updating files, remove compiled caches and re-run:

```powershell
# remove any compiled files
Get-ChildItem -Path . -Recurse -Include *.pyc | Remove-Item -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path . -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# then run the script again
python ./scripts/filter_below_200ma.py --out below_200ma.csv
```

- This is intended as a simple screening tool; please backtest and validate before trading.
- The script uses the NasdaqTrader file `nasdaqlisted.txt` to obtain tickers.
- yfinance may occasionally fail for some tickers; the script retries and skips problematic entries.
- This is intended as a simple screening tool; please backtest and validate before trading.
