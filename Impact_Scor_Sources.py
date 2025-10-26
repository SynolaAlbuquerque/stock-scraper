# --- COLAB-READY TESLA SENTIMENT STOCK PREDICTOR (WITH SOURCE IMPACT ANALYSIS) ---

# 1. Install dependencies
!pip install yfinance textblob vaderSentiment lazypredict scikit-learn pandas numpy matplotlib seaborn --quiet

import re
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from lazypredict.Supervised import LazyRegressor, LazyClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# 2. User Settings
HEADLINES = aditya_headlines_string

KEYWORDS = ['earnings', 'SEC', 'lawsuit', 'investigation', 'FSD', 'recall', 'profit', 'loss', 'battery fire', 'chip shortage','loss', 'drop', 'decline', 'plunge', 'fall', 'slump', 'downturn', 'downgrade',
    'cut', 'reduce', 'shortfall', 'weak', 'slowdown', 'bearish', 'selloff',
    'lawsuit', 'probe', 'investigation', 'recall', 'defect', 'delay', 'accident',
    'fire', 'crash', 'explosion', 'fraud', 'scandal', 'controversy',
    'complaint', 'fine', 'penalty', 'regulatory', 'sec', 'litigation',
    'chip shortage', 'supply issue', 'strike', 'layoff', 'job cuts', 'resignation',
    'competition', 'antitrust', 'inflation', 'interest rate hike', 'cost overrun',
    'bankruptcy', 'default', 'downtime', 'negative outlook','profit', 'growth', 'increase', 'record', 'beat estimates', 'upgrade',
    'surge', 'soar', 'rally', 'bullish', 'expansion', 'rebound', 'strong demand',
    'partnership', 'contract', 'investment', 'approval', 'launch', 'innovation',
    'delivery record', 'autonomy', 'breakthrough', 'efficiency', 'milestone',
    'production ramp', 'earnings beat']
KEYWORD_BOOST = 2.0

USE_3DAYS = True
USE_5DAYS = True
STOCK = 'TSLA'
START_DATE = '2021-01-03'
END_DATE = datetime.datetime.today().strftime('%Y-%m-%d')

# 3. Parse Headlines with Source
def parse_headlines(text):
    lines = text.strip().split('\n')
    records = []
    buff, date, source = "", None, None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if re.match(r'^\d+\.', line):
            if buff and date is not None:
                records.append({'date': date, 'headline': buff.strip(), 'source': source})
            buff = re.sub(r'^\d+\.\s*', '', line).strip()
            date = None
            source = None

        elif line.startswith('Source:'):
            match = re.match(r'Source:\s*(.*)', line)
            if match:
                source = match.group(1).strip()

        elif line.startswith('Published on:'):
            match = re.search(r'(\d{2} \w{3} \d{4})', line)
            if match:
                date = pd.to_datetime(match.group(1), format='%d %b %Y')

        elif not line.startswith('Source:'):
            buff += " " + line.strip()

    if buff and date is not None:
        records.append({'date': date, 'headline': buff.strip(), 'source': source})

    return pd.DataFrame(records)

news_df = parse_headlines(HEADLINES)
news_df = news_df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
print(f"✓ Parsed {len(news_df)} headlines.")

# 4. Sentiment Analysis
analyzer = SentimentIntensityAnalyzer()

def calc_sentiments(row):
    text = row['headline']
    vs = analyzer.polarity_scores(text)
    vader_score = vs['compound']
    tb_score = TextBlob(text).sentiment.polarity
    amplify = any(kw.lower() in text.lower() for kw in KEYWORDS)
    boost = KEYWORD_BOOST if amplify else 1.0
    vader_score *= boost
    tb_score *= boost
    return pd.Series({'vader': vader_score, 'textblob': tb_score})

news_df[['vader', 'textblob']] = news_df.apply(calc_sentiments, axis=1)
news_df['sentiment'] = 0.5 * news_df['vader'] + 0.5 * news_df['textblob']

# 5. Fetch Stock Data
df = yf.download(STOCK, start=START_DATE, end=END_DATE)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

df = df.reset_index()
df['Date'] = pd.to_datetime(df['Date'])

expected_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
if "Adj Close" in df.columns:
    expected_cols.append("Adj Close")
df = df[expected_cols]

# Calculate stock price changes
df = df.rename(columns={'Date': 'date'})
df['date'] = pd.to_datetime(df['date'])
df['pct_change'] = df['Close'].pct_change() * 100
df['next_day_change'] = df['pct_change'].shift(-1)

# ===== SOURCE IMPACT ANALYSIS =====
print("\n" + "="*70)
print("ANALYZING WHICH SOURCES AFFECT STOCK PRICES MORE HEAVILY")
print("="*70)

# Merge news with stock data
news_with_prices = pd.merge(news_df, df[['date', 'Close', 'pct_change', 'next_day_change']], 
                             on='date', how='left')
news_with_prices = news_with_prices.dropna(subset=['next_day_change'])

# Calculate impact metrics per source
source_impact = []

for source in news_with_prices['source'].unique():
    if pd.isna(source):
        continue
    
    source_data = news_with_prices[news_with_prices['source'] == source]
    
    if len(source_data) < 5:  # Need minimum data points
        continue
    
    # Correlation between sentiment and next-day price change
    corr_pearson, p_pearson = pearsonr(source_data['sentiment'], source_data['next_day_change'])
    corr_spearman, p_spearman = spearmanr(source_data['sentiment'], source_data['next_day_change'])
    
    # Average absolute price change following this source's headlines
    avg_abs_change = source_data['next_day_change'].abs().mean()
    
    # Directional accuracy: does sentiment sign match price movement sign?
    sentiment_direction = (source_data['sentiment'] > 0).astype(int)
    price_direction = (source_data['next_day_change'] > 0).astype(int)
    directional_accuracy = (sentiment_direction == price_direction).mean()
    
    # Average sentiment magnitude
    avg_sentiment_magnitude = source_data['sentiment'].abs().mean()
    
    # Weighted impact score (correlation * frequency * avg_change)
    impact_score = abs(corr_pearson) * len(source_data) * avg_abs_change
    
    source_impact.append({
        'source': source,
        'headline_count': len(source_data),
        'pearson_corr': corr_pearson,
        'p_value': p_pearson,
        'spearman_corr': corr_spearman,
        'avg_abs_price_change': avg_abs_change,
        'directional_accuracy': directional_accuracy,
        'avg_sentiment_magnitude': avg_sentiment_magnitude,
        'impact_score': impact_score,
        'significant': 'Yes' if p_pearson < 0.05 else 'No'
    })

impact_df = pd.DataFrame(source_impact)
impact_df = impact_df.sort_values('impact_score', ascending=False).reset_index(drop=True)

print("\n📊 SOURCE IMPACT RANKING (by Impact Score)")
print("Impact Score = |Correlation| × Headline Count × Avg Price Change\n")
display(impact_df)

# Save to CSV
impact_df.to_csv('source_impact_analysis.csv', index=False)
print("\n✓ Detailed analysis saved to 'source_impact_analysis.csv'")

# ===== VISUALIZATIONS =====

# 1. Top Sources by Impact Score
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

top_n = min(15, len(impact_df))
top_sources = impact_df.head(top_n)

# Chart 1: Impact Score
axes[0,0].barh(range(top_n), top_sources['impact_score'], color='darkred', edgecolor='black')
axes[0,0].set_yticks(range(top_n))
axes[0,0].set_yticklabels(top_sources['source'])
axes[0,0].set_xlabel('Impact Score', fontsize=12, fontweight='bold')
axes[0,0].set_title('Top Sources by Stock Price Impact', fontsize=14, fontweight='bold')
axes[0,0].invert_yaxis()
axes[0,0].grid(axis='x', alpha=0.3)

# Chart 2: Correlation Strength
colors = ['green' if x > 0 else 'red' for x in top_sources['pearson_corr']]
axes[0,1].barh(range(top_n), top_sources['pearson_corr'], color=colors, edgecolor='black')
axes[0,1].set_yticks(range(top_n))
axes[0,1].set_yticklabels(top_sources['source'])
axes[0,1].set_xlabel('Pearson Correlation', fontsize=12, fontweight='bold')
axes[0,1].set_title('Sentiment-Price Correlation by Source', fontsize=14, fontweight='bold')
axes[0,1].axvline(x=0, color='black', linewidth=1)
axes[0,1].invert_yaxis()
axes[0,1].grid(axis='x', alpha=0.3)

# Chart 3: Direct
           
