#headlines weighted using vader and textblob and downloaded in csv format using pandas 

pip install newsapi-python
pip install pandas textblob vaderSentiment


from newsapi import NewsApiClient
from datetime import datetime

# Initialize News API client
api_key = "92371a7becd64accb4adb90de19feee2"  # Replace with your actual API key
newsapi = NewsApiClient(api_key=api_key)

# ---- Configuration ----
query = "Tesla"   # The stock or company you're interested in
from_date = "2025-09-25"
to_date = "2025-10-24"
language = "en"

# ---- Fetch News ----
all_articles = newsapi.get_everything(
    q=query,
    from_param=from_date,
    to=to_date,
    language=language,
    sort_by="relevancy",
    page_size=100  # You can increase this up to 100
)

# ---- Display Results ----
print(f"\nTop news headlines about '{query}' from {from_date} to {to_date}:\n")

for i, article in enumerate(all_articles["articles"], start=1):
    title = article["title"]
    source = article["source"]["name"]
    published_at = article["publishedAt"]
    
    # Convert publishedAt to readable format
    readable_time = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ").strftime("%d %b %Y, %I:%M %p")
    
    print(f"{i}. {title}")
    print(f"   Source: {source}")
    print(f"   Published on: {readable_time}\n")

import re
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Example input (paste your text block here)
raw_text = """
"""

# --- Step 1: Extract headlines and dates using regex ---
pattern = r"(\d+)\.\s+(.*?)\n\s+Source: (.*?)\n\s+Published on: (.*?)\n"
matches = re.findall(pattern, raw_text, re.DOTALL)

# --- Step 2: Store parsed data in a list of dicts ---
data = []
for serial, headline, source, published in matches:
    data.append({
        "Serial": int(serial),
        "Headline": headline.strip(),
        "Date": published.strip()
    })

# --- Step 3: Sentiment Analysis ---
vader = SentimentIntensityAnalyzer()

for d in data:
    d["VADER_Compound"] = vader.polarity_scores(d["Headline"])["compound"]
    blob = TextBlob(d["Headline"]).sentiment
    d["TextBlob_Polarity"] = blob.polarity
    d["TextBlob_Subjectivity"] = blob.subjectivity

# --- Step 4: Create DataFrame ---
df = pd.DataFrame(data)

# --- Step 5: Display only Serial & Date, with sentiment scores ---
final_df = df[["Serial", "Date", "VADER_Compound", "TextBlob_Polarity", "TextBlob_Subjectivity"]]

print("\nParsed and Scored Headlines:\n")
print(final_df.to_string(index=False))
final_df.to_csv("headlines_sentiment.csv", index=False)

# Save the DataFrame to a CSV file
final_df.to_csv("headlines_sentiment.csv", index=False)

print("File saved as headlines_sentiment.csv")
from google.colab import files

final_df.to_csv("headlines_sentiment.csv", index=False)
files.download("headlines_sentiment.csv")



