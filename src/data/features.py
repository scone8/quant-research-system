"""
features.py

Processes raw OHLCV Parquet data for S&P 100 tickers and computes features:
- Returns
- Rolling momentum
- Rolling volatility
- Volume z-score

Parquet columns can be multi-index tuples like:
('Date', ''), ('Close', 'TICKER'), ('High', 'TICKER'), ('Low', 'TICKER'),
('Open', 'TICKER'), ('Volume', 'TICKER'), ('Year', '')

Features are computed using 'Close' (no Adj Close needed) and appended
to the original dataframe.
"""

import os
import re
import pandas as pd
from tqdm import tqdm
import logging

# -----------------------------
# CONFIGURATION
# -----------------------------
RAW_DIR = "../../data/raw"
PROCESSED_DIR = "../../data/processed"
LOG_DIR = "../../logs"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "features.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

logging.info("Starting feature engineering...")

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def flatten_columns(df):
    new_cols = []
    for col in df.columns:
        # If it looks like a tuple string "(something, something)", extract the first part
        if col.startswith("(") and "," in col:
            col_str = col.split(",")[0].replace("(", "").replace("'", "").strip()
        else:
            col_str = col
        new_cols.append(col_str)
    
    df.columns = new_cols
    
    # Move 'Date' to the front
    if 'Date' in df.columns:
        cols = df.columns.tolist()
        cols.insert(0, cols.pop(cols.index('Date')))
        df = df[cols]
    
    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def compute_features(df):
    """Compute features and append to original dataframe."""
    df = df.copy()

    # Ensure numeric for calculations
    if 'Close' not in df.columns or 'Volume' not in df.columns:
        logging.error("Missing 'Close' or 'Volume', cannot compute features")
        return df

    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

    # Drop rows with missing critical data
    df.dropna(subset=['Close', 'Volume'], inplace=True)

    # Daily return
    df['Return'] = df['Close'].pct_change()

    # Rolling momentum
    df['Momentum_5'] = df['Close'].pct_change(5)
    df['Momentum_20'] = df['Close'].pct_change(20)
    df['Momentum_60'] = df['Close'].pct_change(60)

    # Rolling volatility
    df['Vol_5'] = df['Return'].rolling(5).std()
    df['Vol_20'] = df['Return'].rolling(20).std()
    df['Vol_60'] = df['Return'].rolling(60).std()

    # Volume z-score
    df['Vol_Z'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()

    # Fill NaNs in features only
    feature_cols = ['Return', 'Momentum_5', 'Momentum_20', 'Momentum_60', 'Vol_5', 'Vol_20', 'Vol_60', 'Vol_Z']
    df[feature_cols] = df[feature_cols].fillna(0)

    return df

# -----------------------------
# PROCESS ALL TICKERS
# -----------------------------
tickers = [d.split("=")[1] for d in os.listdir(RAW_DIR) if d.startswith("ticker=")]

for ticker in tqdm(tickers, desc="Processing tickers"):
    ticker_path = os.path.join(RAW_DIR, f"ticker={ticker}")

    for year_folder in os.listdir(ticker_path):
        year_path = os.path.join(ticker_path, year_folder)
        file_path = os.path.join(year_path, "data.parquet")

        if not os.path.exists(file_path):
            logging.warning(f"{file_path} does not exist, skipping.")
            continue

        try:
            df = pd.read_parquet(file_path)
            df = flatten_columns(df)

            # -----------------------------
            # Compute features and append
            # -----------------------------
            df_features = compute_features(df)

            # -----------------------------
            # Save processed features
            # -----------------------------
            processed_dir = os.path.join(PROCESSED_DIR, f"ticker={ticker}", year_folder)
            ensure_dir(processed_dir)
            processed_path = os.path.join(processed_dir, "features.parquet")

            # Append if file exists and remove duplicates
            if os.path.exists(processed_path):
                old_df = pd.read_parquet(processed_path)
                df_features = pd.concat([old_df, df_features], ignore_index=True)
                df_features.drop_duplicates(subset=['Date'], inplace=True)

            df_features.to_parquet(processed_path, index=False)
            logging.info(f"Saved features for {ticker} {year_folder} to {processed_path}")

        except Exception as e:
            logging.error(f"Error processing {ticker} {year_folder}: {e}")

logging.info("Feature engineering complete!")
