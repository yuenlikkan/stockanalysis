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
from typing import List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm

# Primary source (may be blocked by some networks). We provide fallbacks below.
NASDAQ_SYMBOLS_URLS = [
    "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    # older CSV endpoint (may be slower)
    "https://old.nasdaq.com/screening/companies-by-name.aspx?exchange=NASDAQ&render=download",
    # community mirror on GitHub (fallback when Nasdaq endpoints are unreachable)
    "https://raw.githubusercontent.com/datasets/nasdaq-listings/master/data/nasdaq-listed.csv",
]
CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "nasdaq_listed_cached.csv")


def fetch_nasdaq_symbols() -> List[str]:
    """Download NASDAQ-listed tickers from multiple sources with retries and fallback.

    Tries primary Nasdaq FTP endpoint first, then older Nasdaq CSV endpoint, then a
    GitHub mirror. If a cached file exists locally and all network attempts fail,
    the cache will be used as a last resort.
    """

    def parse_nasdaq_txt_lines(lines: List[str]) -> List[str]:
        syms: List[str] = []
        for line in lines:
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
            syms.append(sym)
        return syms

    # try sources in order
    last_exc = None
    for url in NASDAQ_SYMBOLS_URLS:
        try:
            r = requests.get(url, timeout=(5, 20), headers={"User-Agent": "python-requests/stockanalysis"})
            r.raise_for_status()
            text = r.text
            # GitHub mirror is CSV with header 'Symbol,Security Name'
            if 'Symbol,Security Name' in text.splitlines()[0]:
                # parse CSV
                lines = text.splitlines()
                syms = []
                for i, line in enumerate(lines):
                    if i == 0:
                        continue
                    parts = line.split(",")
                    if parts:
                        syms.append(parts[0].strip())
                # cache
                try:
                    with open(CACHE_PATH, "w", encoding="utf-8") as fh:
                        fh.write(text)
                except Exception:
                    pass
                return syms
            # otherwise assume nasdaqlisted pipe-delimited format
            lines = text.splitlines()
            syms = parse_nasdaq_txt_lines(lines)
            # cache raw text as well
            try:
                with open(CACHE_PATH, "w", encoding="utf-8") as fh:
                    fh.write(text)
            except Exception:
                pass
            return syms
        except Exception as e:
            last_exc = e
            # try next source after a short backoff
            time.sleep(1.0)

    # if network failed, try to load cached copy
    try:
        cache_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "nasdaq_listed_cached.csv"))
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as fh:
                text = fh.read()
            # try CSV parse first
            if 'Symbol,Security Name' in text.splitlines()[0]:
                lines = text.splitlines()
                syms = [line.split(",")[0].strip() for line in lines[1:] if line.strip()]
                return syms
            return parse_nasdaq_txt_lines(text.splitlines())
    except Exception:
        pass

    # if we reach here, raise the last network exception to the caller
    if last_exc:
        raise last_exc
    return []


def chunked(iterable: List[str], size: int):
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


def compute_latest_and_ma200(df: Union[pd.DataFrame, pd.Series, None]) -> Tuple[float, float, Optional[pd.Timestamp]]:
    """Given ticker historical DataFrame or Series, return (latest_close, ma200, ma200_date).

    If there's not enough history to compute 200-day MA, return (latest_close, nan, None).
    Handles Series inputs (interpreted as Close series) and DataFrame inputs.
    """
    if df is None:
        return (math.nan, math.nan, None)

    # If it's a Series, treat it as the Close series
    if isinstance(df, pd.Series):
        close = df.dropna()
    else:
        if df.empty:
            return (math.nan, math.nan, None)
        # ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        # prefer the 'Close' column, otherwise pick the first numeric column
        if "Close" in df.columns:
            close = df["Close"].dropna()
        else:
            numeric_cols = df.select_dtypes(include=["number"]).columns
            if len(numeric_cols) == 0:
                return (math.nan, math.nan, None)
            close = df[numeric_cols[0]].dropna()

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
    parser.add_argument("--threads", action="store_true", help="If set, allow multi-threaded yfinance downloads (bool) - off by default for reliability")
    parser.add_argument("--top", type=int, default=0, help="If >0, restrict to top N tickers by market cap before screening")
    parser.add_argument("--index", choices=["nasdaq100", "dow"], default="nasdaq100", help="Index to screen: 'nasdaq100' (default) or 'dow'")
    args = parser.parse_args(argv)

    # Determine history period: we need at least 250 trading days; download 320 calendar days to be safe
    end = dt.date.today()
    start = None
    if args.start:
        start = dt.datetime.strptime(args.start, "%Y-%m-%d").date()
    else:
        start = end - dt.timedelta(days=400)

    # Choose index
    if args.index == "nasdaq100":
        print("Fetching Nasdaq-100 constituents from Wikipedia (default)...")
    else:
        print("Fetching Dow constituents from Wikipedia...")

    def fetch_nasdaq100_symbols() -> List[str]:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0 Safari/537.36"}

        cachef = os.path.join(os.path.dirname(__file__), "..", "nasdaq100_cached.csv")

        # Fetch HTML via requests to avoid 403; if network fails, fall back to cache
        try:
            r = requests.get(url, timeout=(5, 20), headers=headers)
            r.raise_for_status()
            html = r.text
        except Exception:
            if os.path.exists(cachef):
                with open(cachef, "r", encoding="utf-8") as fh:
                    return [r.strip() for r in fh.read().splitlines() if r.strip()]
            raise

        # Parse HTML into tables
        try:
            tables = pd.read_html(html)
        except ImportError as ie:
            raise RuntimeError("Missing HTML parser dependency: install 'lxml' (pip install lxml)") from ie
        except Exception:
            # parsing failed; try cache
            if os.path.exists(cachef):
                with open(cachef, "r", encoding="utf-8") as fh:
                    return [r.strip() for r in fh.read().splitlines() if r.strip()]
            raise

        # find table with a 'Ticker' or 'Ticker symbol' column
        for t in tables:
            cols = [c.lower() for c in t.columns.astype(str)]
            if any("ticker" in c for c in cols):
                # pick the column name that contains 'ticker'
                ticker_col = [c for c in t.columns.astype(str) if "ticker" in c.lower()][0]
                syms = [str(x).strip() for x in t[ticker_col].tolist()]
                # cache
                try:
                    with open(cachef, "w", encoding="utf-8") as fh:
                        fh.write("\n".join(syms))
                except Exception:
                    pass
                return syms

        raise RuntimeError("Could not find Nasdaq-100 table on Wikipedia page")

    def fetch_dow_symbols() -> List[str]:
        # Dow Jones Industrial Average constituents page
        url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0 Safari/537.36"}
        cachef = os.path.join(os.path.dirname(__file__), "..", "dow_cached.csv")
        try:
            r = requests.get(url, timeout=(5, 20), headers=headers)
            r.raise_for_status()
            html = r.text
        except Exception:
            if os.path.exists(cachef):
                with open(cachef, "r", encoding="utf-8") as fh:
                    return [r.strip() for r in fh.read().splitlines() if r.strip()]
            raise

        try:
            tables = pd.read_html(html)
        except ImportError as ie:
            raise RuntimeError("Missing HTML parser dependency: install 'lxml' (pip install lxml)") from ie
        except Exception:
            if os.path.exists(cachef):
                with open(cachef, "r", encoding="utf-8") as fh:
                    return [r.strip() for r in fh.read().splitlines() if r.strip()]
            raise

        # Wikipedia page has a table of DJIA constituents — find the ticker column
        for t in tables:
            cols = [c.lower() for c in t.columns.astype(str)]
            if any("symbol" in c or "ticker" in c for c in cols):
                ticker_col = [c for c in t.columns.astype(str) if ("symbol" in c.lower() or "ticker" in c.lower())][0]
                syms = [str(x).strip() for x in t[ticker_col].tolist()]
                try:
                    with open(cachef, "w", encoding="utf-8") as fh:
                        fh.write("\n".join(syms))
                except Exception:
                    pass
                return syms
        raise RuntimeError("Could not find Dow constituents table on Wikipedia page")

    try:
        if args.index == "nasdaq100":
            symbols = fetch_nasdaq100_symbols()
        else:
            symbols = fetch_dow_symbols()
    except Exception as e:
        print(f"Failed to fetch symbols: {type(e).__name__}: {e}")
        return 1
    print(f"Using {len(symbols)} symbols from {args.index}")

    # optionally restrict to top N by market cap
    if args.top and args.top > 0:
        print(f"Fetching market caps to determine top {args.top} tickers...")
        # fetch market caps in chunks to avoid overloading yfinance
        market_caps = {}
        for batch in chunked(symbols, max(50, args.chunk_size)):
            try:
                tickers_str = " ".join(batch)
                tickers = yf.Tickers(tickers_str)
                # Tickers.tickers is a dict mapping symbol->Ticker
                for sym, tk in tickers.tickers.items():
                    try:
                        info = tk.info
                        mc = info.get("marketCap") or info.get("market_cap") or 0
                        market_caps[sym] = mc if isinstance(mc, (int, float)) else 0
                    except Exception:
                        market_caps[sym] = 0
            except Exception:
                # best-effort: set zeros for this batch
                for s in batch:
                    market_caps[s] = 0
            time.sleep(0.1)
        # sort symbols by market cap desc and take top N
        symbols = [s for s, _ in sorted(market_caps.items(), key=lambda kv: kv[1], reverse=True)][: args.top]
        print(f"Selected top {len(symbols)} tickers by market cap")

    # Prepare output
    out_rows = []
    failed_tickers = []

    total = len(symbols)
    print(f"Got {total} symbols. Downloading historical data in chunks of {args.chunk_size}...")

    # Fetch per-ticker sequentially. Batch downloads using yfinance can hang in some
    # environments, so we prefer a robust per-ticker approach with retries.
    for batch in tqdm(list(chunked(symbols, args.chunk_size)), desc="batches"):
        for ticker in batch:
            for attempt in range(3):
                try:
                    print(f"Downloading ticker: {ticker} (attempt {attempt+1}/3)")
                    df = yf.download(
                        tickers=ticker,
                        start=start.isoformat(),
                        end=end.isoformat(),
                        progress=False,
                        threads=False,
                        group_by="ticker",
                        auto_adjust=False,
                        prepost=False,
                    )
                    # yfinance may return an empty DataFrame on some errors; check contents
                    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                        raise RuntimeError("Empty result from yf.download")
                    break
                except Exception as e:
                    msg = str(e)
                    # If we see Yahoo 'Invalid Crumb' / 401 errors, try Ticker.history fallback
                    if "Invalid Crumb" in msg or "Unauthorized" in msg or "401" in msg:
                        try:
                            print(f"yfinance download returned 401/crumb error for {ticker}; trying Ticker.history() fallback")
                            tk = yf.Ticker(ticker)
                            df = tk.history(start=start.isoformat(), end=end.isoformat(), actions=False)
                            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                                raise RuntimeError("Empty result from Ticker.history fallback")
                            break
                        except Exception:
                            # fallback failed; treat as a normal retry
                            pass
                    wait = 1 + attempt * 2
                    print(f"yfinance download error for {ticker} (attempt {attempt+1}/3): {type(e).__name__}: {msg[:200]}... Retrying in {wait}s...")
                    time.sleep(wait)
            else:
                print(f"Failed to download {ticker} after retries; marking as failed")
                failed_tickers.append(ticker)
                continue

            # compute on the returned frame/series
            latest_close, ma200, ma200_date = compute_latest_and_ma200(df)
            if math.isnan(latest_close) and math.isnan(ma200):
                failed_tickers.append(ticker)
                continue
            if not math.isnan(ma200) and latest_close < ma200:
                out_rows.append((ticker, latest_close, ma200, ma200_date.date() if ma200_date is not None else ""))
            # small pause to be polite to the service
            time.sleep(0.1)

    # Save results
    out_path = os.path.abspath(args.out)
    print(f"Writing {len(out_rows)} results to {out_path}")
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["ticker", "latest_close", "ma200", "ma200_date"])
        for r in out_rows:
            writer.writerow(r)

    if failed_tickers:
        failed_path = os.path.abspath(os.path.join(os.path.dirname(out_path), "failed_downloads.csv"))
        print(f"Writing {len(failed_tickers)} failed tickers to {failed_path}")
        with open(failed_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["ticker"])
            for t in sorted(set(failed_tickers)):
                w.writerow([t])

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
