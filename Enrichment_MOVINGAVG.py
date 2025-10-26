# ================== STEP 16: FEATURE ENRICHMENT & LIGHTGBM MODEL ==================
!pip install lightgbm --quiet
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- Copy and enrich from your merged dataframe ---
df = full.copy()

# 1. Enrich with stock-based features
df['daily_return'] = df['Close'].pct_change() * 100
df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
df['volatility_3d'] = df['daily_return'].rolling(3).std()
df['volatility_5d'] = df['daily_return'].rolling(5).std()
df['volatility_10d'] = df['daily_return'].rolling(10).std()
df['volume_change'] = df['Volume'].pct_change()
df['volume_ma_5d'] = df['Volume'].rolling(5).mean()
df['price_ma_5d'] = df['Close'].rolling(5).mean()
df['price_ma_10d'] = df['Close'].rolling(10).mean()
df['price_ma_ratio'] = df['Close'] / df['price_ma_5d']

# 2. Sentiment lags
df['lag_sentiment_1d'] = df['sentiment'].shift(1)
df['lag_sentiment_3d'] = df['sentiment'].shift(3)
df['lag_sentiment_5d'] = df['sentiment'].shift(5)
df['lag_sentiment_avg_3d'] = df['sent_3d'].shift(1)
df['lag_sentiment_avg_5d'] = df['sent_5d'].shift(1)

# 3. Interaction terms
df['sentiment_x_return'] = df['sentiment'] * df['daily_return']
df['sentiment_x_vol'] = df['sentiment'] * df['volatility_3d']

# 4. Drop NaNs from rolling/lag ops
df = df.dropna().reset_index(drop=True)

# 5. Define features and target
features = [
    'sentiment', 'sent_3d', 'sent_5d',
    'lag_sentiment_1d', 'lag_sentiment_3d', 'lag_sentiment_5d',
    'daily_return', 'log_return',
    'volatility_3d', 'volatility_5d', 'volatility_10d',
    'volume_change', 'volume_ma_5d',
    'price_ma_5d', 'price_ma_10d', 'price_ma_ratio',
    'sentiment_x_return', 'sentiment_x_vol'
]
target = 'Close'

# 6. Train/Test Split (time-ordered)
split_idx = int(0.8 * len(df))
X_train, X_test = df[features].iloc[:split_idx], df[features].iloc[split_idx:]
y_train, y_test = df[target].iloc[:split_idx], df[target].iloc[split_idx:]

# 7. LightGBM Model + parameter tuning
params = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
}
lgbm = LGBMRegressor(**params)
lgbm.fit(X_train, y_train)

# 8. Predictions & metrics
y_pred = lgbm.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"✅ LightGBM RMSE: {rmse:.4f}")
print(f"✅ LightGBM R²: {r2:.4f}")

# 9. Feature Importance
importances = pd.Series(lgbm.feature_importances_, index=features).sort_values(ascending=False)
plt.figure(figsize=(8,6))
sns.barplot(x=importances, y=importances.index, palette="viridis")
plt.title("Feature Importances (LightGBM)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

# 10. Actual vs Predicted Prices
plt.figure(figsize=(10,5))
plt.plot(df['date'].iloc[split_idx:], y_test.values, label='Actual', color='blue')
plt.plot(df['date'].iloc[split_idx:], y_pred, label='Predicted', color='orange')
plt.title(f"{STOCK} Price Prediction with Enriched Features (LightGBM)")
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print("📈 Feature-enriched LightGBM analysis complete.")
