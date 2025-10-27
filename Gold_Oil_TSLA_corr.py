# --- TESLA, GOLD, AND CRUDE OIL CORRELATION ANALYSIS (FULLY ROBUST VERSION) ---

# 1. Install dependencies
!pip install yfinance pandas numpy matplotlib seaborn --quiet

# 2. Import libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 3. Define tickers
tickers = {
    "TSLA": "Tesla",
    "GC=F": "Gold",
    "CL=F": "Crude_Oil"
}

# 4. Download data
raw_data = yf.download(list(tickers.keys()), start="2021-01-04", end="2025-10-27")

# 5. Extract usable price data robustly
if isinstance(raw_data.columns, pd.MultiIndex):
    # If multi-index, try 'Adj Close', otherwise fall back to 'Close'
    if 'Adj Close' in raw_data.columns.levels[0]:
        data = raw_data['Adj Close'].copy()
    elif 'Close' in raw_data.columns.levels[0]:
        data = raw_data['Close'].copy()
    else:
        raise KeyError("No 'Adj Close' or 'Close' column found in Yahoo Finance data.")
else:
    # Flat columns — directly use what’s available
    data = raw_data.copy()

# Rename columns for readability
data.rename(columns=tickers, inplace=True)

# --- Diagnostics before alignment ---
print("Raw data shape:", data.shape)
print("Missing values per asset (raw):")
print(data.isna().sum())
print()

# Align to business days (B) to get a common calendar for pairwise comparisons
# This will create rows for all business days in the date range
data_b = data.asfreq('B')

# Forward-fill then back-fill as a sensible imputation for market-close series
# Forward-fill uses last known price; backfill fills leading NaNs (if any) from first available
data_ffill = data_b.ffill().bfill()

# Show if anything still missing
print("After asfreq + ffill/bfill, missing per asset:")
print(data_ffill.isna().sum())
print()

# Compute daily percent returns on the aligned/fill data
returns = data_ffill.pct_change().dropna(how='all')

# How many overlapping samples pairwise? We'll print a pairwise count matrix
def pairwise_overlap_counts(df):
    cols = df.columns
    n = len(cols)
    mat = pd.DataFrame(np.zeros((n,n), dtype=int), index=cols, columns=cols)
    for i in range(n):
        for j in range(n):
            # count rows where both series are finite
            a = df.iloc[:, i]
            b = df.iloc[:, j]
            mat.iloc[i,j] = int(((~a.isna()) & (~b.isna())).sum())
    return mat

overlap_counts = pairwise_overlap_counts(returns)
print("Pairwise overlapping samples (# days where both returns available):")
print(overlap_counts)
print()

# Compute correlation matrix using a minimum overlapping-sample threshold.
# Set min_periods to e.g. 30 to avoid spurious correlations from tiny overlaps.
MIN_OVERLAP = 30
corr_matrix = returns.corr(min_periods=MIN_OVERLAP)

print("=== Correlation Matrix (Daily Returns) with min_periods={} ===".format(MIN_OVERLAP))
print(corr_matrix, "\n")

# Plot correlation heatmap (mask any NaN)
plt.figure(figsize=(6,4))
sns.heatmap(corr_matrix.fillna(0), annot=True, cmap="coolwarm", fmt=".2f",
            vmin=-1, vmax=1, linewidths=.5)
plt.title("Correlation Between Tesla, Gold, and Crude Oil (Daily Returns)")
plt.show()

# Normalized price comparison (use the aligned & ffilled price data)
plt.figure(figsize=(10,6))
(data_ffill / data_ffill.iloc[0]).plot(ax=plt.gca())
plt.title("Normalized Price Comparison: Tesla vs Gold vs Crude Oil (Aligned business days)")
plt.xlabel("Date")
plt.ylabel("Normalized Price (Start = 1)")
plt.legend()
plt.grid(True)
plt.show()
