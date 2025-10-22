pip install yfinance
import yfinance as yf
data = yf.download("GC=F", period="6mo", interval="1d")
print(data.head())
data.to_csv("USD_INR_data.csv")
from google.colab import files
files.download("USD_INR_data.csv")
