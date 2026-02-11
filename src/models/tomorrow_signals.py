import pandas as pd
import joblib
import os

# Load ML model
model = joblib.load("../../models/xgb_momentum_model.joblib")

# Load latest features for all tickers
processed_dir = "../../data/processed"
tickers = [d.split("=")[1] for d in os.listdir(processed_dir) if d.startswith("ticker=")]
features_list = []

for ticker in tickers:
    ticker_dir = os.path.join(processed_dir, f"ticker={ticker}")
    # Get latest year folder
    year_folder = sorted(os.listdir(ticker_dir))[-1]
    df = pd.read_parquet(os.path.join(ticker_dir, year_folder, "features.parquet"))
    latest = df.sort_values("Date").iloc[-1].copy()
    latest['Ticker'] = ticker
    features_list.append(latest)

latest_features = pd.DataFrame(features_list)
feature_cols = ['Momentum_5','Momentum_20','Momentum_60','Vol_5','Vol_20','Vol_60','Vol_Z']

# Predict probability of positive return tomorrow
latest_features['ML_Prob'] = model.predict_proba(latest_features[feature_cols])[:,1]

# Generate binary signal (1=buy, 0=hold)
latest_features['ML_Signal'] = (latest_features['ML_Prob'] > 0.5).astype(int)

# Save tomorrow's signals
latest_features[['Ticker','ML_Prob','ML_Signal']].to_csv(
    "../../data/processed/ml_signals/tomorrow_signals.csv", index=False
)

print(latest_features[['Ticker','ML_Prob','ML_Signal']])
