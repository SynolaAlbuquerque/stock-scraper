pip install yfinance
import yfinance as yf
data = yf.download("AAPL", period="6mo", interval="1d")
print(data.head())
data.to_csv("apple_stock_data.csv")
from google.colab import files
files.download("apple_stock_data.csv")
