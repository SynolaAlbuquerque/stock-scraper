# Stock Sentiment Prediction — Tesla Case Study

This project analyzes how **financial news sentiment** affects **Tesla (TSLA)** stock price movements using Natural Language Processing (NLP), feature engineering, and machine learning models (LightGBM).

The pipeline fetches news data, computes sentiment scores, enriches them with temporal features, merges them with market data, and trains both **regression** and **classification models** to predict stock behavior.

---

## Overview

* **Goal:** Quantify how sentiment from financial headlines correlates with Tesla’s stock movements.
* **Approach:**

  1. Collect real news headlines (from APIs or text files).
  2. Assign sentiment scores using **VADER** and **TextBlob**.
  3. Merge with **historical stock prices** using `yfinance`.
  4. Engineer rolling and source-level sentiment features.
  5. Train:

     * **Regression model:** Predict next-day percentage change.
     * **Classification model:** Predict direction (Up/Down).
  6. Evaluate and visualize model performance.
  7. Identify which **keywords** most influence stock movement.

---

## ⚙️ Requirements

Install dependencies:

```bash
pip install yfinance vaderSentiment textblob scikit-learn matplotlib pandas numpy seaborn lightgbm
```

---

## 1. Collecting News Headlines

Fetch or import Tesla-related headlines and store them in a DataFrame:

```python
news_df = pd.read_csv("tesla_headlines.csv")
```

Each entry includes:

* `date`
* `headline`
* `source` (optional)

---

## 2. Sentiment Scoring

Apply **VADER** and **TextBlob** to compute sentiment polarity:

```python
news_df['vader'] = news_df['headline'].apply(lambda x: sia.polarity_scores(x)['compound'])
news_df['textblob'] = news_df['headline'].apply(lambda x: TextBlob(x).sentiment.polarity)
news_df['sentiment'] = (news_df['vader'] + news_df['textblob']) / 2
```

---

## 3. Stock Data Integration

Use Yahoo Finance to fetch Tesla stock data:

```python
import yfinance as yf
stock_df = yf.download("TSLA", start="2021-01-04", end="2025-10-27", interval="1d")
```

Merge daily stock prices with average daily sentiment by date.

---

## 4. Feature Engineering

Create rolling features and sentiment aggregates:

* **Rolling Sentiment Averages:** `sent_3d`, `sent_5d`
* **Source-Level Averages:** `sent_Reuters`, `sent_Bloomberg`, etc.
* **Average Source Impact:** Weight for how much each source influences sentiment trend.

These features capture **short-term sentiment momentum** and **source credibility**.

---

## 5. Regression Model — Predicting Next-Day Price Change

Predicts the **next day’s % change** in Tesla’s closing price.

### Model Setup

```python
lgbm = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
lgbm.fit(X_train, y_train)
```

### Evaluation

* **RMSE (Root Mean Squared Error):** Prediction accuracy in same units as % change.
* **R² Score:** Explains how much variance the model accounts for.

**Example Output:**

```
✅ LightGBM RMSE: 2.5342
✅ LightGBM R²: 0.72
```

### Visualizations

* **Feature Importance:** Shows which features drive predictions.
* **Actual vs Predicted Prices:** Visual comparison of model tracking vs true data.

---

## 6. Classification Model — Predicting Up/Down Movement

Reframes regression into a binary problem:

> Predict whether the next day’s price will rise (1) or fall (0).

### Model Setup

```python
clf = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
clf.fit(X_train_cls, y_train_cls)
```

### Evaluation Metrics

* **Accuracy Score**
* **Precision / Recall / F1-Score**
* **Confusion Matrix (Custom Colored)**
* **ROC Curve & AUC**

**Example Output:**

```
✅ Classification Accuracy: 0.714
📊 Classification Report:
              precision    recall  f1-score   support
           0      0.69      0.72      0.70       102
           1      0.74      0.71      0.73       120
```

**Visualization:**

* 🟩 *Green cells* → correct predictions
* 🟥 *Red cells* → misclassifications
* 📈 ROC curve shows model’s discrimination power.

---

## 7. Keyword Impact Analysis

Quantifies which **keywords** in headlines most influence Tesla’s **price change** and **direction**.

### Regression Keyword Impact

Calculates correlation between sentiment on a keyword and **next-day % change**.

```python
impact_score = |correlation| × avg(abs(price change)) × frequency
```

| keyword  | pearson_corr | avg_abs_price_change | impact_score | significant |
| -------- | ------------ | -------------------- | ------------ | ----------- |
| earnings | 0.28         | 3.6                  | 18.2         | ✅           |
| recall   | -0.33        | 4.1                  | 21.0         | ✅           |
| lawsuit  | -0.25        | 3.1                  | 15.4         | ⚠️          |

### Classification Keyword Impact

Measures how strongly sentiment around a keyword predicts **up vs down days**.

| keyword     | occurrences | avg_sentiment | pct_up_days | corr_sent_vs_dir | impact_score |
| ----------- | ----------- | ------------- | ----------- | ---------------- | ------------ |
| earnings    | 84          | 0.22          | 0.68        | 0.31             | 26.0         |
| partnership | 15          | 0.34          | 0.80        | 0.40             | 6.0          |
| lawsuit     | 31          | -0.18         | 0.33        | -0.29            | 9.0          |
| recall      | 42          | -0.15         | 0.40        | -0.25            | 10.5         |

### Insights

* **Positive-impact keywords:** *earnings*, *partnership*, *delivery*
* **Negative-impact keywords:** *recall*, *lawsuit*, *crash*
* These terms often precede **predictable market reactions**.

---

## 8. Visual Outputs

| Plot                             | Description                           |
| -------------------------------- | ------------------------------------- |
| `feature_importance.png`         | Relative contribution of each feature |
| `actual_vs_predicted_prices.png` | Regression performance over time      |
| `confusion_matrix_manual.png`    | Classification results (Up/Down)      |
| `roc_curve.png`                  | ROC–AUC for classifier                |
| `keyword_impact.png`             | Most influential keywords             |

---
## 📂 Project Structure

```
📁 stock-sentiment-tesla/
├── data/
│   ├── tesla_headlines.csv
│   └── stock_data.csv
├── notebooks/
│   └── sentiment_pipeline.ipynb
├── models/
│   ├── regression_lightgbm.pkl
│   └── classification_lightgbm.pkl
├── visuals/
│   ├── feature_importance.png
│   ├── actual_vs_predicted_prices.png
│   ├── confusion_matrix_manual.png
│   ├── roc_curve.png
│   └── keyword_impact.png
├── keyword_impact_analysis.py
├── regression_model.py
├── classification_model.py
└── README.md
```

---
