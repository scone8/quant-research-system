"""
ingest.py

Pulls historical OHLCV data for S&P 100 tickers,
saves partitioned Parquet files locally and optionally uploads to S3.

Features:
- Partitioned Parquet storage
- Error handling with logging
- Progress tracking
- Optional S3 integration
"""

import os
import pandas as pd
import yfinance as yf
from tqdm import tqdm
import logging
#import boto3
#from botocore.exceptions import NoCredentialsError

# -----------------------------
# CONFIGURATION
# -----------------------------
RAW_DIR = "../../data/raw"
UNIVERSE_FILE = "../../data/universe/sp100.csv"
LOG_DIR = "../../logs"
os.makedirs(LOG_DIR, exist_ok=True)
BENCHMARK_TICKERS = ["SPY"]
ONLY_BENCHMARK = os.getenv("INGEST_ONLY_BENCHMARK", "0") == "1"

USE_S3 = False
S3_BUCKET = "your-bucket-name"
S3_PREFIX = "quant-project/raw"

START_YEAR = 2008
END_YEAR = 2023

# -----------------------------
# SETUP LOGGING
# -----------------------------
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "ingest.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

logging.info("Starting data ingestion...")

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def upload_to_s3(local_path, s3_path):
    s3 = boto3.client("s3")
    try:
        s3.upload_file(local_path, S3_BUCKET, s3_path)
        logging.info(f"Uploaded to S3: {s3_path}")
    except NoCredentialsError:
        logging.warning("AWS credentials not found. Skipping S3 upload.")

# -----------------------------
# LOAD TICKERS
# -----------------------------
try:
    tickers_df = pd.read_csv(UNIVERSE_FILE)
    universe_tickers = tickers_df['Ticker'].dropna().astype(str).str.upper().tolist()
    if ONLY_BENCHMARK:
        tickers = BENCHMARK_TICKERS.copy()
    else:
        tickers = list(dict.fromkeys(universe_tickers + BENCHMARK_TICKERS))
    logging.info(f"Loaded {len(tickers)} tickers from {UNIVERSE_FILE}")
except Exception as e:
    logging.error(f"Failed to load tickers: {e}")
    raise

# -----------------------------
# INGEST DATA
# -----------------------------
for ticker in tqdm(tickers, desc="Downloading tickers"):
    try:
        df = yf.download(
            ticker,
            start=f"{START_YEAR}-01-01",
            end=f"{END_YEAR}-12-31",
            progress=False
        )
        if df.empty:
            logging.warning(f"No data for {ticker}, skipping.")
            continue

        df.reset_index(inplace=True)
        df['Year'] = df['Date'].dt.year

        for year, year_df in df.groupby('Year'):
            year_dir = os.path.join(RAW_DIR, f"ticker={ticker}", f"year={year}")
            ensure_dir(year_dir)
            file_path = os.path.join(year_dir, "data.parquet")
            year_df.to_parquet(file_path, engine='pyarrow', index=False)
            logging.info(f"Saved {ticker} {year} to {file_path}")

            if USE_S3:
                s3_path = f"{S3_PREFIX}/ticker={ticker}/year={year}/data.parquet"
                upload_to_s3(file_path, s3_path)

    except Exception as e:
        logging.error(f"Error downloading {ticker}: {e}")

logging.info("Data ingestion complete!")
