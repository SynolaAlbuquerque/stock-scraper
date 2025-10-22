import requests as req
from bs4 import BeautifulSoup as BS
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# download VADER sentiment lexicon
nltk.download('vader_lexicon')

# create sentiment analyzer
sia = SentimentIntensityAnalyzer()

url = "https://www.businesstoday.in/latest/economy"
webpage = req.get(url)
trav = BS(webpage.content, "html.parser")

headlines = []  # to store (headline, sentiment) pairs

M = 1
for link in trav.find_all('a'):
    if (str(type(link.string)) == "<class 'bs4.element.NavigableString'>"
        and link.string is not None
        and len(link.string) > 35):

        text = link.string.strip()
        sentiment = sia.polarity_scores(text)

        print(f"{M}. {text}")
        print(f"   → Sentiment: {sentiment}\n")

        headlines.append({
            "headline": text,
            "pos": sentiment["pos"],
            "neu": sentiment["neu"],
            "neg": sentiment["neg"],
            "compound": sentiment["compound"]
        })

        M += 1
#compound score is the main thing we will be looking at... values > 0.05 → Positive, < -0.05 → Negative, Between -0.05 and +0.05 → Neutral
