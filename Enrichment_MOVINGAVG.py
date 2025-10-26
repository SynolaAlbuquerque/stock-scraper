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

# ================== STEP 17: CLASSIFICATION EVALUATION ON ENRICHED MODEL ==================
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# 1. Define classification target: next-day up/down direction
df['next_close'] = df['Close'].shift(-1)
df['next_change'] = df['next_close'] - df['Close']
df['direction'] = (df['next_change'] > 0).astype(int)

# 2. Align features (drop last NaN after shift)
df_cls = df.dropna().reset_index(drop=True)
X_cls = df_cls[features]
y_cls = df_cls['direction']

# 3. Train/test split (time-ordered)
split_idx = int(0.8 * len(df_cls))
X_train_cls, X_test_cls = X_cls.iloc[:split_idx], X_cls.iloc[split_idx:]
y_train_cls, y_test_cls = y_cls.iloc[:split_idx], y_cls.iloc[split_idx:]

# 4. Train a LightGBMClassifier with same hyperparams
from lightgbm import LGBMClassifier
clf_params = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}
clf = LGBMClassifier(**clf_params)
clf.fit(X_train_cls, y_train_cls)

# 5. Predictions & evaluation
y_pred_cls = clf.predict(X_test_cls)
y_pred_proba = clf.predict_proba(X_test_cls)[:,1]

acc = accuracy_score(y_test_cls, y_pred_cls)
cm = confusion_matrix(y_test_cls, y_pred_cls)
report = classification_report(y_test_cls, y_pred_cls, digits=3)

print(f"✅ Classification Accuracy: {acc:.3f}")
print("\n📊 Classification Report:\n", report)

# 6. Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cbar=False, cmap='viridis', linewidths=1, linecolor='black')
plt.title('LightGBM Classification Confusion Matrix', fontsize=14)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 7. ROC Curve
fpr, tpr, _ = roc_curve(y_test_cls, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='orange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0,1], [0,1], color='blue', linestyle='--')
plt.title('ROC Curve (Up vs Down)', fontsize=14)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print("📈 Classification evaluation complete.")
