import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Config ---
tickers = {"TSLA": "Tesla", "GC=F": "Gold", "CL=F": "Crude_Oil"}
START, END = "2021-01-04", "2025-10-27"
RESAMPLE = "D"   # or 'W' for weekly, 'M' for monthly
MIN_OVERLAP = 30

# --- Download & Extract Adjusted Close ---
raw = yf.download(list(tickers.keys()), start=START, end=END)
if isinstance(raw.columns, pd.MultiIndex):
    data = raw["Adj Close"] if "Adj Close" in raw.columns.levels[0] else raw["Close"]
else:
    data = raw.copy()
data.rename(columns=tickers, inplace=True)

# --- Align and Clean ---
data = data.asfreq('B').ffill().bfill()

# --- Compute Log Returns ---
returns = np.log(data / data.shift(1)).dropna()
if RESAMPLE != "D":
    returns = returns.resample(RESAMPLE).sum()

# --- Correlation ---
corr_matrix = returns.corr(min_periods=MIN_OVERLAP)

# --- Display ---
print(f"=== {RESAMPLE}-Return Correlation Matrix ===")
print(corr_matrix, "\n")

# --- Heatmap ---
plt.figure(figsize=(6,4))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title(f"{RESAMPLE}-Return Correlation: Tesla, Gold, Crude Oil")
plt.tight_layout()
plt.show()

# --- Rolling Correlation (Tesla vs others) ---
window = 90  # 3-month rolling window
rolling_corr = returns["Tesla"].rolling(window).corr(returns[["Gold", "Crude_Oil"]])

plt.figure(figsize=(9,5))
rolling_corr.plot(ax=plt.gca())
plt.title(f"{window}-Day Rolling Correlation with Tesla")
plt.ylabel("Correlation")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Normalized Prices ---
(data / data.iloc[0]).plot(figsize=(10,6))
plt.title("Normalized Price Comparison (Start = 1)")
plt.ylabel("Normalized Price")
plt.grid(True)
plt.tight_layout()
plt.show()
