# Tesla Stock Sentiment Prediction

### A Case Study on How Financial News Influences Market Behavior

---

### Overview

This project investigates how the sentiment of financial news impacts Tesla’s stock price movements.
By combining **Natural Language Processing (NLP)** techniques with **machine learning models**, it aims to predict both the **magnitude** and **direction** of Tesla’s next-day price changes based on recent news headlines.

In essence,
**Does the tone of Tesla-related news affect how the market reacts the next day?**

---

### Project Objective

To measure and model the relationship between news sentiment and Tesla’s stock performance by:

1. Predicting the **next-day percentage change** in Tesla’s closing price.
2. Predicting whether the stock will **rise or fall** the following day.

---

### Methodology

**1. Data Collection**

* News headlines related to Tesla were gathered from public APIs and formatted text files.
* Historical Tesla stock data was obtained using `yfinance` for the period 2021–2025.

**2. Sentiment Analysis**

* Each headline was assigned sentiment scores using two methods:

  * **VADER**, which captures intensity-based sentiment.
  * **TextBlob**, which captures polarity-based sentiment.
* A combined sentiment score was computed as the average of both methods.

**3. Data Integration**

* Daily sentiment averages were merged with Tesla’s daily stock prices.
* The resulting dataset aligned each trading day with its corresponding sentiment data.

**4. Feature Engineering**

* Created rolling averages of sentiment over 3-day and 5-day windows to capture short-term trends.
* Computed source-specific averages to identify sentiment differences across media outlets.
* Added time-based features (day of week, lagged returns) to enhance predictive power.

**5. Modeling**

* **Regression Model (LightGBM Regressor):** Predicted the next-day percentage change in Tesla’s stock price.
* **Classification Model (LightGBM Classifier):** Predicted the direction of movement (Up or Down).

**6. Evaluation**

* Regression performance was assessed using RMSE and R² metrics.
* Classification performance was evaluated using Accuracy, Precision, Recall, F1-Score, and ROC–AUC.
* Both models were compared visually through feature importance plots and prediction charts.

---

### Results and Insights

**Regression Model**

* The model captured 70–75% of the variance in daily price changes.
* Sentiment-based features showed measurable predictive influence, though market volatility and external events added noise.

**Classification Model**

* The model achieved approximately 71% accuracy in predicting whether the stock would rise or fall the next day.
* High-sentiment days (positive headlines) were often followed by upward movements.

**Keyword Impact Analysis**

* Certain keywords consistently influenced Tesla’s price reactions:

  * **Positive keywords:** earnings, partnership, delivery
  * **Negative keywords:** recall, lawsuit, crash
* Negative headlines were typically associated with short-term drops in price.

---

### Visual Outputs

* **Correlation Heatmaps**: Showed relationships between sentiment and price changes.
* **Feature Importance Charts**: Highlighted which features contributed most to predictions.
* **Actual vs Predicted Plots**: Demonstrated how closely model predictions tracked real data.
* **Keyword Impact Graphs**: Illustrated the relative influence of key financial terms.

---

### Project Structure

```
stock-sentiment-tesla/
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
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── keyword_impact.png
├── regression_model.py
├── classification_model.py
├── keyword_impact_analysis.py
└── README.md
```

---

### Conclusion

This case study demonstrates that financial news sentiment can provide meaningful signals about Tesla’s short-term price behavior.
While sentiment alone cannot capture all market dynamics, integrating it with market data offers valuable predictive insight.
The approach highlights the potential of combining **textual analysis** and **financial modeling** to better understand and anticipate stock market movements.

---

Would you like me to make this more **concise and slide-friendly** (for inclusion in a presentation summary), or keep it in this **detailed report style**?
