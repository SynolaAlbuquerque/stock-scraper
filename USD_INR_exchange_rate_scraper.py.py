!pip install yfinance --quiet
import yfinance as yf
from google.colab import files
data = yf.download("INR=X", start="2021-01-04", end="2025-10-27", interval="1d")
print(data.head())
data.to_csv("USD_INR_data.csv")
files.download("USD_INR_data.csv")

