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
import io
import yfinance as yf
from tqdm import tqdm
import os as _os

# Default cache directory for fetched histories
CACHE_DIR_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "data_cache")

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
            # If multiple numeric columns exist, pick the first; if the DataFrame
            # has a single column, squeeze it to a Series to allow scalar access.
            numeric_cols = df.select_dtypes(include=["number"]).columns
            if len(numeric_cols) == 0:
                return (math.nan, math.nan, None)
            close = df[numeric_cols[0]].dropna()

    # If close is still a DataFrame (e.g., single-column DF slice), convert to Series
    if isinstance(close, pd.DataFrame):
        # Try to squeeze single-column DataFrame to Series safely
        squeezed = close.squeeze(axis=1)
        if isinstance(squeezed, pd.Series):
            close = squeezed.dropna()
        else:
            # pick first numeric column
            numeric_cols = close.select_dtypes(include=["number"]).columns
            if len(numeric_cols) == 0:
                return (math.nan, math.nan, None)
            close = close[numeric_cols[0]].dropna()

    if close.empty:
        return (math.nan, math.nan, None)

    # ensure close is a Series (squeeze if necessary)
    if isinstance(close, pd.DataFrame):
        if close.shape[1] >= 1:
            close = close.iloc[:, 0]
        else:
            return (math.nan, math.nan, None)

    # now close should be a Series; get latest value and MA
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
    parser.add_argument("--max-pe", type=float, default=50.0, help="Only include tickers with trailing P/E ratio below this value (default 50). Set 0 to disable.)")
    parser.add_argument("--index", choices=["nasdaq100", "dow"], default="nasdaq100", help="Index to screen: 'nasdaq100' (default) or 'dow'")
    parser.add_argument("--cache-dir", default=None, help="Directory to store per-ticker cached history CSVs (default: data_cache)")
    parser.add_argument("--refresh-cache", action="store_true", help="Ignore cache and force fresh network fetches for histories")
    args = parser.parse_args(argv)

    # Try to force yfinance to use curl_cffi backend when available which can
    # improve cookie/crumb handling in some environments.
    try:
        _os.environ.setdefault("YF_USE_CURL_CFFI", "1")
    except Exception:
        pass

    # Optionally force using Stooq only (no yfinance) by setting environment
    # variable YF_FORCE_STOOQ=1. This is useful when Yahoo returns authorization
    # errors in the current environment.
    FORCE_STOOQ = str(_os.environ.get("YF_FORCE_STOOQ", "0")) == "1"

    # Cache directory setup: per-ticker CSVs stored here
    cache_dir = args.cache_dir or CACHE_DIR_DEFAULT
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    REFRESH_CACHE = bool(args.refresh_cache)

    def cache_path_for(ticker: str) -> str:
        # use lowercase and replace slashes/dots to safe filenames
        fn = ticker.lower().replace('/', '_').replace('\\', '_').replace('.', '-').replace('^', '') + '.csv'
        return os.path.join(cache_dir, fn)

    def is_cache_fresh(ticker: str) -> bool:
        if REFRESH_CACHE:
            return False
        p = cache_path_for(ticker)
        if not os.path.exists(p):
            return False
        # consider cache fresh if modified on same calendar date (local time)
        mtime = dt.datetime.fromtimestamp(os.path.getmtime(p)).date()
        return mtime == dt.date.today()

    def load_cache(ticker: str) -> Optional[pd.DataFrame]:
        p = cache_path_for(ticker)
        try:
            if os.path.exists(p):
                df = pd.read_csv(p, parse_dates=[0])
                if df.empty:
                    return None
                df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
                df.set_index('Date', inplace=True)
                if 'Close' not in df.columns and 'close' in df.columns:
                    df.rename(columns={'close': 'Close'}, inplace=True)
                return df
        except Exception:
            return None
        return None

    def save_cache(ticker: str, df: pd.DataFrame) -> None:
        try:
            p = cache_path_for(ticker)
            # ensure Date index becomes first column
            out = df.copy()
            out = out.reset_index()
            out.to_csv(p, index=False)
        except Exception:
            pass

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
            # pd.read_html warns when passing a literal string; wrap in StringIO
            tables = pd.read_html(io.StringIO(html))
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
            # pd.read_html warns when passing a literal string; wrap in StringIO
            tables = pd.read_html(io.StringIO(html))
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

    def prepare_yf_session(ticker: str) -> Optional[object]:
        """Return a requests-like session primed with Yahoo Finance cookies/headers.

        This performs a lightweight GET to the Yahoo Finance quote page to populate
        cookies/crumbs that yfinance can reuse. Returns None on failure.
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        # Prefer curl_cffi requests session if available (handles curl-style requests)
        try:
            import curl_cffi.requests as curl_requests  # type: ignore
            session = curl_requests.Session()
        except Exception:
            session = requests.Session()
        try:
            # Use the ticker's Yahoo quote page to get cookies/crumbs
            url = f"https://finance.yahoo.com/quote/{ticker}"
            session.headers.update(headers)
            session.get(url, timeout=(5, 15))
            return session
        except Exception:
            try:
                # fallback: try without raising
                return session
            except Exception:
                return None

    def fetch_stooq_history(ticker: str) -> Optional[pd.DataFrame]:
        """Fetch daily history from Stooq for a US ticker. Returns DataFrame or None.

        Stooq uses lowercase with hyphens for dots, e.g. BRK.B -> brk-b.us
        URL: https://stooq.com/q/d/l/?s={symbol}.us&i=d
        """
        # Use a small retry loop and avoid streaming reads which can hang
        sym = ticker.lower().replace('.', '-').replace('^', '')
        url = f"https://stooq.com/q/d/l/?s={sym}.us&i=d"
        headers = {"User-Agent": "python-requests/stockanalysis"}
        # If cache is fresh, load from disk and avoid network
        try:
            if is_cache_fresh(ticker):
                cached = load_cache(ticker)
                if cached is not None:
                    return cached
        except Exception:
            # ignore cache errors
            pass

        # Create a session with retries to handle transient network issues more robustly
        session = requests.Session()
        try:
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry

            retry_strategy = Retry(
                total=3,
                backoff_factor=0.5,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=("GET",),
                raise_on_status=False,
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
        except Exception:
            # If retry imports fail, continue with plain session
            pass

        attempts = 3
        for attempt in range(1, attempts + 1):
            try:
                r = session.get(url, timeout=(5, 8), headers=headers, allow_redirects=True)
                r.raise_for_status()
                text = r.text
                if not text or not text.strip():
                    return None
                df = pd.read_csv(io.StringIO(text), parse_dates=[0])
                if df.empty:
                    return None
                df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
                df.set_index('Date', inplace=True)
                if 'Close' not in df.columns and 'close' in df.columns:
                    df.rename(columns={'close': 'Close'}, inplace=True)
                # save to cache for reuse during the day
                try:
                    save_cache(ticker, df)
                except Exception:
                    pass
                return df
            except (requests.exceptions.RequestException, ConnectionError, OSError) as rexc:
                # transient network error; retry with backoff
                if attempt < attempts:
                    time.sleep(0.5 * attempt)
                    continue
                return None
            except (pd.errors.EmptyDataError, ValueError):
                return None
            except Exception:
                return None
        try:
            session.close()
        except Exception:
            pass
        return None

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
        # fetch market caps per-ticker (avoid yf.Tickers aggregator which may trigger
        # larger aggregated requests and 401s). If we detect authorization (401)
        # errors from Yahoo, skip top-N selection and continue with full symbol list.
        market_caps: dict[str, float] = {}
        saw_auth_error = False
        for s in symbols:
            try:
                # try a couple times per ticker
                mc_val = 0
                for info_try in range(2):
                    try:
                        tk = yf.Ticker(s)
                        info = tk.info or {}
                        mc = info.get("marketCap") or info.get("market_cap") or 0
                        if isinstance(mc, (int, float)):
                            mc_val = mc
                        break
                    except Exception as ie:
                        msg = str(ie)
                        if "Invalid Crumb" in msg or "401" in msg or "Unauthorized" in msg:
                            saw_auth_error = True
                            break
                        time.sleep(0.2)
                market_caps[s] = mc_val
            except Exception:
                market_caps[s] = 0
            # be polite
            time.sleep(0.05)

        if saw_auth_error:
            print("Detected Yahoo authorization errors while fetching market caps. Skipping --top selection and continuing with full symbol list.")
        else:
            # sort symbols by market cap desc and take top N
            symbols = [s for s, _ in sorted(market_caps.items(), key=lambda kv: kv[1], reverse=True)][: args.top]
            print(f"Selected top {len(symbols)} tickers by market cap")

    # Prepare output
    out_rows = []
    failed_tickers = []
    # cache for ticker info (marketCap, trailingPE, etc.) to reduce repeated info calls
    info_cache: dict[str, dict] = {}

    total = len(symbols)
    print(f"Got {total} symbols. Downloading historical data in chunks of {args.chunk_size}...")

    # Use batched downloads to reduce the number of HTTP requests to Yahoo and
    # improve reliability. We'll download batches with yf.download and then
    # inspect returned DataFrame (grouped by ticker when multiple tickers are
    # provided). Only after a ticker passes the price < MA test will we fetch
    # additional info (P/E) to keep info calls minimal.
    for batch in tqdm(list(chunked(symbols, args.chunk_size)), desc="batches"):
        tickers_str = " ".join(batch)
        # Attempt to download the batch with retries
        # If FORCE_STOOQ is enabled, fetch per-ticker from Stooq to avoid Yahoo entirely.
        if FORCE_STOOQ:
            # build a mapping ticker->DataFrame using Stooq
            data_map: dict[str, pd.DataFrame] = {}
            for ticker in batch:
                st_df = fetch_stooq_history(ticker)
                if st_df is not None:
                    data_map[ticker] = st_df
                else:
                    data_map[ticker] = pd.DataFrame()
            data = data_map
        else:
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
                    print(f"yfinance download error (attempt {attempt+1}/3): {type(e).__name__}: {str(e)[:200]}... Retrying in {wait}s...")
                    time.sleep(wait)
            else:
                print(f"Failed to download batch: {batch[:3]}... marking all as failed")
                failed_tickers.extend(batch)
                continue

    # yfinance returns a DataFrame with columns as top-level if multiple tickers
        if isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex):
            # iterate tickers
            for ticker in batch:
                try:
                    df = data[ticker].dropna(how="all")
                except Exception:
                    df = pd.DataFrame()
                # If yfinance returned empty, try Stooq fallback
                if (isinstance(df, pd.DataFrame) and df.empty) or df is None:
                    stooq_df = fetch_stooq_history(ticker)
                    if stooq_df is not None:
                        df = stooq_df

                latest_close, ma200, ma200_date = compute_latest_and_ma200(df)
                if math.isnan(latest_close) and math.isnan(ma200):
                    failed_tickers.append(ticker)
                    continue

                # apply price < ma200 filter first
                if not math.isnan(ma200) and latest_close < ma200:
                    pe_val = None
                    # Fetch P/E only after ticker passes price<MA to reduce .info calls
                    if args.max_pe and args.max_pe > 0:
                        if ticker in info_cache:
                            info = info_cache[ticker]
                        else:
                            info = {}
                            try:
                                tk = yf.Ticker(ticker)
                                info = tk.info or {}
                            except Exception:
                                info = {}
                            info_cache[ticker] = info
                        if isinstance(info, dict):
                            pe_val = info.get("trailingPE") or info.get("trailing_pe") or info.get("priceToEarnings")
                            try:
                                if pe_val is not None:
                                    pe_val = float(pe_val)
                            except Exception:
                                pe_val = None

                    if args.max_pe and args.max_pe > 0:
                        if pe_val is None:
                            # missing PE -> exclude
                            continue
                        if pe_val < float(args.max_pe):
                            out_rows.append((ticker, latest_close, ma200, ma200_date.date() if ma200_date is not None else "", pe_val))
                    else:
                        out_rows.append((ticker, latest_close, ma200, ma200_date.date() if ma200_date is not None else "", pe_val))
    else:
            # handle single-ticker DataFrame shapes
            if isinstance(data, pd.DataFrame) and "Close" in data.columns:
                # assume single ticker in this batch
                ticker = batch[0]
                df = data
                if (isinstance(df, pd.DataFrame) and df.empty) or df is None:
                    stooq_df = fetch_stooq_history(ticker)
                    if stooq_df is not None:
                        df = stooq_df
                latest_close, ma200, ma200_date = compute_latest_and_ma200(df)
                if math.isnan(latest_close) and math.isnan(ma200):
                    failed_tickers.append(ticker)
                else:
                    pe_val = None
                    if not math.isnan(ma200) and latest_close < ma200:
                        if args.max_pe and args.max_pe > 0:
                            try:
                                tk = yf.Ticker(ticker)
                                info = tk.info or {}
                            except Exception:
                                info = {}
                            pe_val = info.get("trailingPE") if isinstance(info, dict) else None
                            try:
                                if pe_val is not None:
                                    pe_val = float(pe_val)
                            except Exception:
                                pe_val = None
                            if pe_val is None:
                                # missing PE -> exclude
                                pass
                            elif pe_val < float(args.max_pe):
                                out_rows.append((ticker, latest_close, ma200, ma200_date.date() if ma200_date is not None else "", pe_val))
                        else:
                            out_rows.append((ticker, latest_close, ma200, ma200_date.date() if ma200_date is not None else "", pe_val))
            # After each batch, flush partial results to disk so progress is saved
            try:
                tmp_out = os.path.abspath(args.out)
                with open(tmp_out, "w", newline="", encoding="utf-8") as fh:
                    writer = csv.writer(fh)
                    writer.writerow(["ticker", "latest_close", "ma200", "ma200_date", "pe"])
                    for r in out_rows:
                        writer.writerow(r)
                if failed_tickers:
                    failed_path = os.path.abspath(os.path.join(os.path.dirname(tmp_out), "failed_downloads.csv"))
                    with open(failed_path, "w", newline="", encoding="utf-8") as fh:
                        w = csv.writer(fh)
                        w.writerow(["ticker"])
                        for t in sorted(set(failed_tickers)):
                            w.writerow([t])
            except Exception:
                # best-effort flush; failures here are not fatal
                pass
            else:
                # Unexpected format: fallback to per-ticker downloads for resilience
                for ticker in batch:
                    try:
                        df = yf.download(ticker, start=start.isoformat(), end=end.isoformat(), progress=False, threads=bool(args.threads))
                    except Exception:
                        df = pd.DataFrame()
                    if (isinstance(df, pd.DataFrame) and df.empty) or df is None:
                        stooq_df = fetch_stooq_history(ticker)
                        if stooq_df is not None:
                            df = stooq_df
                    latest_close, ma200, ma200_date = compute_latest_and_ma200(df)
                    if math.isnan(latest_close) and math.isnan(ma200):
                        failed_tickers.append(ticker)
                        continue
                    pe_val = None
                    if not math.isnan(ma200) and latest_close < ma200:
                        if args.max_pe and args.max_pe > 0:
                            try:
                                tk = yf.Ticker(ticker)
                                info = tk.info or {}
                            except Exception:
                                info = {}
                            pe_val = info.get("trailingPE") if isinstance(info, dict) else None
                            try:
                                if pe_val is not None:
                                    pe_val = float(pe_val)
                            except Exception:
                                pe_val = None
                            if pe_val is None:
                                continue
                            if pe_val < float(args.max_pe):
                                out_rows.append((ticker, latest_close, ma200, ma200_date.date() if ma200_date is not None else "", pe_val))
                        else:
                            out_rows.append((ticker, latest_close, ma200, ma200_date.date() if ma200_date is not None else "", pe_val))

    # Save results
    out_path = os.path.abspath(args.out)
    print(f"Writing {len(out_rows)} results to {out_path}")
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["ticker", "latest_close", "ma200", "ma200_date", "pe"])
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
