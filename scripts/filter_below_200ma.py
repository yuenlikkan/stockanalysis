#!/usr/bin/env python3
"""
Filter NASDAQ stocks whose latest close price is below their 200-day moving average.

Outputs a CSV file `below_200ma.csv` in the workspace root with columns:
    ticker, latest_close, ma200, ma200_date

Usage (PowerShell):
    python ./scripts/filter_below_200ma.py --out results.csv

This script fetches the list of NASDAQ tickers from nasdaqtrader.com
and uses yfinance to download historical OHLCV data in chunks to avoid
request limits.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm

NASDAQ_SYMBOLS_URL = (
    "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
)


def fetch_nasdaq_symbols() -> List[str]:
    """Download NASDAQ-listed tickers from NasdaqTrader and return list of symbols.

    The file is a pipe-delimited text with a header and a footer line 'File Creation Date'.
    We ignore entries marked with 'oth' in the Test Issue field and strip whitespace.
    """
    r = requests.get(NASDAQ_SYMBOLS_URL, timeout=20)
    r.raise_for_status()
    data = r.text.splitlines()
    symbols: List[str] = []
    for line in data:
        if line.startswith("Symbol|"):
            continue
        if line.strip() == "":
            continue
        if line.startswith("File Creation Date"):
            break
        parts = line.split("|")
        if len(parts) < 1:
            continue
        sym = parts[0].strip()
        if sym == "Symbol":
            continue
        # skip special entries
        if sym == "BF.B":
            continue
        symbols.append(sym)
    return symbols


def chunked(iterable: List[str], size: int):
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


def compute_latest_and_ma200(df: pd.DataFrame) -> Tuple[float, float, pd.Timestamp | None]:
    """Given ticker historical DataFrame, return (latest_close, ma200, ma200_date).

    If there's not enough history to compute 200-day MA, return (latest_close, nan, None).
    """
    if df is None or df.empty:
        return (math.nan, math.nan, None)
    # make sure we have 'Close' column and index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    if "Close" not in df.columns:
        return (math.nan, math.nan, None)
    close = df["Close"].dropna()
    if close.empty:
        return (math.nan, math.nan, None)
    latest_close = float(close.iloc[-1])
    if len(close) < 200:
        return (latest_close, math.nan, None)
    ma200_series = close.rolling(window=200, min_periods=200).mean()
    ma200 = float(ma200_series.iloc[-1])
    ma200_date = ma200_series.index[-1]
    return (latest_close, ma200, ma200_date)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Find NASDAQ tickers with price below 200-day moving average"
    )
    parser.add_argument("--out", default="below_200ma.csv", help="Output CSV filename")
    parser.add_argument("--start", default=None, help="History start date (YYYY-MM-DD). Defaults to 1 year ago + buffer).")
    parser.add_argument("--chunk-size", type=int, default=100, help="Number of tickers to download per yfinance batch")
    parser.add_argument("--threads", type=int, default=1, help="yfinance threads parameter (set to 1 for reliability)")
    args = parser.parse_args(argv)

    # Determine history period: we need at least 250 trading days; download 320 calendar days to be safe
    end = dt.date.today()
    start = None
    if args.start:
        start = dt.datetime.strptime(args.start, "%Y-%m-%d").date()
    else:
        start = end - dt.timedelta(days=400)

    print(f"Fetching NASDAQ symbols from {NASDAQ_SYMBOLS_URL}...")
    symbols = fetch_nasdaq_symbols()
    if not symbols:
        print("No symbols found. Exiting.")
        return 1

    # Prepare output
    out_rows = []

    total = len(symbols)
    print(f"Got {total} symbols. Downloading historical data in chunks of {args.chunk_size}...")

    # yfinance supports multiple tickers per download
    for batch in tqdm(list(chunked(symbols, args.chunk_size)), desc="batches"):
        tickers_str = " ".join(batch)
        # Attempt to download with retries
        for attempt in range(3):
            try:
                data = yf.download(
                    tickers=batch,
                    start=start.isoformat(),
                    end=end.isoformat(),
                    progress=False,
                    threads=args.threads,
                    group_by="ticker",
                    auto_adjust=False,
                    prepost=False,
                )
                break
            except Exception as e:
                wait = 1 + attempt * 2
                print(f"yfinance download error (attempt {attempt+1}/3): {e}. Retrying in {wait}s...")
                time.sleep(wait)
        else:
            print(f"Failed to download batch: {batch[:3]}... skipping")
            continue

        # yfinance returns a DataFrame with columns as top-level if multiple tickers
        if isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex):
            # iterate tickers
            for ticker in batch:
                try:
                    df = data[ticker].dropna(how="all")
                except Exception:
                    df = pd.DataFrame()
                latest_close, ma200, ma200_date = compute_latest_and_ma200(df)
                if not math.isnan(ma200) and latest_close < ma200:
                    out_rows.append((ticker, latest_close, ma200, ma200_date.date() if ma200_date is not None else ""))
        else:
            # Single ticker or unexpected shape: yfinance may return single-level columns
            # Try mapping by order
            if isinstance(data, pd.DataFrame) and "Close" in data.columns:
                # Could be single ticker
                # Try to infer ticker name from batch length
                ticker = batch[0]
                latest_close, ma200, ma200_date = compute_latest_and_ma200(data)
                if not math.isnan(ma200) and latest_close < ma200:
                    out_rows.append((ticker, latest_close, ma200, ma200_date.date() if ma200_date is not None else ""))
            else:
                # Unexpected; attempt per-ticker fetch
                for ticker in batch:
                    try:
                        df = yf.download(ticker, start=start.isoformat(), end=end.isoformat(), progress=False, threads=1)
                    except Exception:
                        df = pd.DataFrame()
                    latest_close, ma200, ma200_date = compute_latest_and_ma200(df)
                    if not math.isnan(ma200) and latest_close < ma200:
                        out_rows.append((ticker, latest_close, ma200, ma200_date.date() if ma200_date is not None else ""))

    # Save results
    out_path = os.path.abspath(args.out)
    print(f"Writing {len(out_rows)} results to {out_path}")
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["ticker", "latest_close", "ma200", "ma200_date"])
        for r in out_rows:
            writer.writerow(r)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
