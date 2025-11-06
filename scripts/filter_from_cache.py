#!/usr/bin/env python3
"""Compute latest close and 200-day MA from per-ticker cache CSVs.

This reads all CSV files in a cache directory (default ../data_cache), computes
the latest close and 200-day moving average, filters tickers where latest < ma200,
and writes an output CSV with columns: ticker,latest_close,ma200,ma200_date.

Usage (PowerShell):
    python ./scripts/filter_from_cache.py --cache-dir ./data_cache --out confirm_run_results.csv
"""
from __future__ import annotations

import argparse
import os
import math
import pandas as pd
import csv


def compute_from_cache(cache_dir: str, out_path: str, ma_window: int = 200) -> int:
    cache_dir = os.path.abspath(cache_dir)
    if not os.path.isdir(cache_dir):
        print(f"Cache directory not found: {cache_dir}")
        return 1

    files = [f for f in os.listdir(cache_dir) if f.lower().endswith('.csv')]
    out_rows = []

    for fn in sorted(files):
        ticker_raw = os.path.splitext(fn)[0]
        # map filename back to ticker: replace '-' with '.' (best-effort)
        ticker = ticker_raw.replace('-', '.').upper()
        p = os.path.join(cache_dir, fn)
        try:
            df = pd.read_csv(p, parse_dates=[0])
            if df.empty:
                continue
            # ensure Date index
            df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
            df.set_index('Date', inplace=True)
            # prefer Close (case-insensitive)
            close_col = None
            for c in df.columns:
                if c.lower() == 'close':
                    close_col = c
                    break
            if close_col is None:
                # try any numeric column
                nums = df.select_dtypes(include=['number']).columns
                if len(nums) == 0:
                    continue
                close_col = nums[0]
            close = df[close_col].dropna()
            if close.empty or len(close) < ma_window:
                continue
            latest_close = float(close.iloc[-1])
            ma = float(close.rolling(window=ma_window, min_periods=ma_window).mean().iloc[-1])
            ma_date = close.rolling(window=ma_window, min_periods=ma_window).mean().index[-1].date()
            if not math.isnan(ma) and latest_close < ma:
                out_rows.append((ticker, latest_close, ma, ma_date))
        except Exception:
            # ignore problematic cache files
            continue

    # write results
    with open(out_path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['ticker', 'latest_close', 'ma200', 'ma200_date'])
        for r in out_rows:
            w.writerow(r)

    print(f"Wrote {len(out_rows)} results to {out_path}")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', default=os.path.join(os.path.dirname(__file__), '..', 'data_cache'))
    parser.add_argument('--out', default='confirm_run_results.csv')
    parser.add_argument('--ma-window', type=int, default=200)
    args = parser.parse_args()
    return compute_from_cache(args.cache_dir, args.out, ma_window=args.ma_window)


if __name__ == '__main__':
    raise SystemExit(main())
