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

# 6. Clean data (remove missing values)
data = data.dropna()

# 7. Calculate daily returns
returns = data.pct_change().dropna()

# 8. Compute correlation matrix
corr_matrix = returns.corr()

print("=== Correlation Matrix (Daily Returns) ===")
print(corr_matrix, "\n")

# 9. Plot correlation heatmap
plt.figure(figsize=(6,4))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between TSLA, Gold, and Crude Oil (Daily Returns)")
plt.show()

# 10. Plot normalized price trends for comparison
plt.figure(figsize=(10,6))
(data / data.iloc[0]).plot(ax=plt.gca())
plt.title("Normalized Price Comparison: Tesla vs Gold vs Crude Oil")
plt.xlabel("Date")
plt.ylabel("Normalized Price (Start = 1)")
plt.legend()
plt.grid(True)
plt.show()
