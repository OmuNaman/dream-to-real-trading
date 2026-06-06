"""Fetch the real-arm equity universe LOCALLY (residential IP; latest yfinance) and save the Close
price matrix as a parquet, to be uploaded to the Modal volume (cloud IPs are blocked by Yahoo/Stooq).
Run: py -3.13 fetch_real.py   ->  writes real_cache_stooq.parquet (matches the cache path _real_arm reads)."""
import yfinance as yf
import pandas as pd

# ~90 liquid large-cap US tickers (a credible pooled universe; survivorship NOT controlled -> optimistic bound)
TICKERS = ("AAPL MSFT AMZN GOOGL META NVDA TSLA JPM V JNJ WMT PG XOM HD CVX MRK ABBV KO PEP BAC "
           "PFE COST DIS CSCO ACN MCD ABT CRM TMO LIN ADBE NKE TXN NEE DHR WFC PM UNP MS BMY RTX "
           "LOW HON AMGN INTC QCOM CAT IBM GS BA SBUX BLK GE AXP BKNG MDT GILD ADP TJX VRTX C SCHW "
           "MMM CB MO DE LMT ADI PLD SYK MDLZ CI SO DUK BDX ICE WM WMB ITW WELL WBA GM F EMR ".split())

print(f"fetching {len(TICKERS)} tickers (yfinance {yf.__version__}) ...")
raw = yf.download(list(TICKERS), start="2010-01-01", end="2024-12-31", auto_adjust=True, progress=False)
# robust Close extraction
if "Close" in raw.columns.get_level_values(0):
    px = raw["Close"]
else:
    px = raw.xs("Close", axis=1, level=-1)
px = px.dropna(how="all")
# keep tickers with a long history
good = [c for c in px.columns if px[c].dropna().shape[0] > 2500]
px = px[good]
print(f"kept {px.shape[1]} tickers x {px.shape[0]} rows; range {px.index.min().date()}..{px.index.max().date()}")
px.to_parquet("real_cache_stooq.parquet")
print("wrote real_cache_stooq.parquet")
