"""
train.py

Train an ML model to predict next-day positive returns for S&P 100 tickers.
Uses features computed in features.py.

Model: XGBoost
Output: predicted signals per ticker per day (1=buy, 0=hold)
"""

import os
import pandas as pd
import numpy as np
import joblib
import logging
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score

# -----------------------------
# CONFIGURATION
# -----------------------------
PROCESSED_DIR = "../../data/processed"
MODEL_DIR = "../../models"
LOG_DIR = "../../logs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

TARGET_COL = "Target"  # next day positive return
FEATURE_COLS = [
    'Momentum_5', 'Momentum_20', 'Momentum_60',
    'Vol_5', 'Vol_20', 'Vol_60',
    'Vol_Z'
]

# -----------------------------
# SETUP LOGGING
# -----------------------------
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "train.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)
logging.info("Starting ML training...")

# -----------------------------
# LOAD FEATURES
# -----------------------------
tickers = [d.split("=")[1] for d in os.listdir(PROCESSED_DIR) if d.startswith("ticker=")]
all_data = []

for ticker in tqdm(tickers, desc="Loading features"):
    ticker_dir = os.path.join(PROCESSED_DIR, f"ticker={ticker}")
    for year_folder in os.listdir(ticker_dir):
        file_path = os.path.join(ticker_dir, year_folder, "features.parquet")
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            df['Ticker'] = ticker
            # Create target: next day positive return
            df[TARGET_COL] = (df['Return'].shift(-1) > 0).astype(int)
            df.dropna(subset=FEATURE_COLS + [TARGET_COL], inplace=True)
            all_data.append(df)

if not all_data:
    logging.error("No data found. Exiting.")
    exit()

data = pd.concat(all_data).sort_values("Date")

X = data[FEATURE_COLS]
y = data[TARGET_COL]

logging.info(f"Loaded {len(data)} rows for ML training.")

# -----------------------------
# TIME-SERIES SPLIT
# -----------------------------
tscv = TimeSeriesSplit(n_splits=5)
fold = 1
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    logging.info(f"Fold {fold} - Accuracy: {acc:.3f}, ROC AUC: {auc:.3f}")
    fold += 1

# -----------------------------
# TRAIN FINAL MODEL ON ALL DATA
# -----------------------------
final_model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss'
)
final_model.fit(X, y)

# Save model
model_path = os.path.join(MODEL_DIR, "xgb_momentum_model.joblib")
joblib.dump(final_model, model_path)
logging.info(f"Final model saved to {model_path}")

# -----------------------------
# GENERATE PREDICTION SIGNALS
# -----------------------------
data['ML_Signal'] = final_model.predict(X)

# Save signals to Parquet for engine.py
signals_dir = os.path.join(PROCESSED_DIR, "ml_signals")
os.makedirs(signals_dir, exist_ok=True)
signals_path = os.path.join(signals_dir, "predicted_signals.parquet")
data[['Date', 'Ticker', 'ML_Signal']].to_parquet(signals_path, index=False)
logging.info(f"Predicted signals saved to {signals_path}")
logging.info("ML training complete!")
