"""
engine.py

Backtesting engine for hybrid ML + momentum strategy
- Thresholdless momentum signals (optionally hybrid ML)
- Uses 2% capital per trade
- Simulates transaction costs and slippage
- Portfolio starts at $1,000
- Equity curve in dollars with matplotlib and ASCII output
"""

import os
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import subprocess
import joblib
import matplotlib.ticker as mtick

# -----------------------------
# CONFIGURATION
# -----------------------------
PROCESSED_DIR = "../../data/processed"
LOG_DIR = "../../logs"
MODEL_FILE = "../../models/xgb_momentum_model.joblib"

USE_ML_SIGNALS = True
INITIAL_CAPITAL = 1_000
RISK_PER_TRADE = 0.02  # 2% of capital per trade
TRANSACTION_COST = 0.001
SLIPPAGE = 0.0005
BACKTEST_START = "2009-01-01"

os.makedirs(LOG_DIR, exist_ok=True)

# -----------------------------
# LOGGING
# -----------------------------
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "backtest.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)
logging.info("Starting backtest...")

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def load_features(ticker):
    ticker_dir = os.path.join(PROCESSED_DIR, f"ticker={ticker}")
    if not os.path.exists(ticker_dir):
        return pd.DataFrame()
    dfs = []
    for year_folder in os.listdir(ticker_dir):
        file_path = os.path.join(ticker_dir, year_folder, "features.parquet")
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            df['Ticker'] = ticker
            dfs.append(df)
    return pd.concat(dfs).sort_values("Date") if dfs else pd.DataFrame()

# -----------------------------
# LOAD DATA
# -----------------------------
tickers = [d.split("=")[1] for d in os.listdir(PROCESSED_DIR) if d.startswith("ticker=")]
all_data = []

for ticker in tqdm(tickers, desc="Loading features"):
    df = load_features(ticker)
    if not df.empty:
        df = df.sort_values("Date")
        df['Return'] = df['Close'].pct_change().fillna(0)
        df['Momentum_5'] = df['Close'].pct_change(5)
        df['Momentum_20'] = df['Close'].pct_change(20)
        df['Momentum_60'] = df['Close'].pct_change(60)
        df['Vol_5'] = df['Return'].rolling(5).std()
        df['Vol_20'] = df['Return'].rolling(20).std()
        df['Vol_60'] = df['Return'].rolling(60).std()
        df['Vol_Z'] = (df['Vol_5'] - df['Vol_20']) / df['Vol_60'].replace(0, np.nan)
        df = df.fillna(0)
        all_data.append(df)

if not all_data:
    logging.error("No data found. Exiting.")
    exit()

data = pd.concat(all_data).sort_values("Date")
data = data[data['Date'] >= BACKTEST_START]
logging.info(f"Loaded features for {len(tickers)} tickers starting from {BACKTEST_START}")

# -----------------------------
# LOAD ML MODEL
# -----------------------------
if USE_ML_SIGNALS:
    if not os.path.exists(MODEL_FILE):
        logging.info("ML model missing, running train.py...")
        subprocess.run(["python", "../models/train.py"], check=True)
    model = joblib.load(MODEL_FILE)
    logging.info("Loaded ML model.")

    feature_cols = ['Momentum_5','Momentum_20','Momentum_60','Vol_5','Vol_20','Vol_60','Vol_Z']
    X = data[feature_cols]
    data['ML_Prob'] = model.predict_proba(X)[:,1]
    logging.info("ML probabilities calculated.")

# -----------------------------
# THRESHOLDLESS SIGNAL
# -----------------------------
data['Signal'] = (data['Momentum_5'] > 0).astype(int)

# -----------------------------
# REINDEX TO FULL DATE RANGE
# -----------------------------
all_dates = pd.date_range(data['Date'].min(), data['Date'].max())
signals_dict = {}
returns_dict = {}

for ticker in tickers:
    df = data[data['Ticker'] == ticker].set_index('Date')
    signals_dict[ticker] = df['Signal'].reindex(all_dates, fill_value=0)
    returns_dict[ticker] = df['Return'].reindex(all_dates, fill_value=0)

signals = pd.DataFrame(signals_dict, index=all_dates)
returns = pd.DataFrame(returns_dict, index=all_dates)
logging.info(f"Backtesting from {signals.index.min().date()} to {signals.index.max().date()}")

# -----------------------------
# BACKTEST PORTFOLIO - 2% RISK PER TRADE (CORRECTED)
# -----------------------------
portfolio_value = INITIAL_CAPITAL
portfolio_history = []

for date in all_dates:
    daily_signals = signals.loc[date]
    daily_returns = returns.loc[date]

    active_tickers = daily_signals[daily_signals > 0].index
    num_active = len(active_tickers)

    if num_active == 0:
        portfolio_history.append(portfolio_value)
        continue

    # Each trade risks 2% of current portfolio
    dollar_per_trade = portfolio_value * RISK_PER_TRADE

    # Daily P&L in dollars
    daily_pnl = sum(dollar_per_trade * daily_returns[ticker] for ticker in active_tickers)

    # Transaction cost per trade
    cost = (TRANSACTION_COST + SLIPPAGE) * dollar_per_trade * num_active

    # Update portfolio
    portfolio_value += daily_pnl - cost
    portfolio_history.append(portfolio_value)

portfolio_history = pd.Series(portfolio_history, index=all_dates)





# -----------------------------
# PERFORMANCE METRICS
# -----------------------------
def sharpe_ratio(returns, risk_free=0.0):
    return (returns.mean() - risk_free) / returns.std() * (252 ** 0.5)

def max_drawdown(portfolio):
    cum_max = portfolio.cummax()
    drawdown = (portfolio - cum_max) / cum_max
    return drawdown.min()

daily_returns_series = portfolio_history.pct_change().fillna(0)
sharpe = sharpe_ratio(daily_returns_series)
drawdown = max_drawdown(portfolio_history)
cagr = (portfolio_history.iloc[-1] / INITIAL_CAPITAL) ** (1 / (len(portfolio_history)/252)) - 1

logging.info(f"CAGR: {cagr:.2%}")
logging.info(f"Sharpe Ratio: {sharpe:.2f}")
logging.info(f"Max Drawdown: {drawdown:.2%}")


# -----------------------------
# ASCII EQUITY CURVE
# -----------------------------
def ascii_plot(series, width=80, height=20):
    min_val = series.min()
    max_val = series.max()
    scale = (max_val - min_val) / height if max_val != min_val else 1
    blocks = [" " * width for _ in range(height)]

    step = max(1, len(series) // width)
    sampled = series[::step].values
    plot_width = len(sampled)

    for i, val in enumerate(sampled):
        level = int((val - min_val) / scale)
        level = min(level, height - 1)
        row = list(blocks[height - level - 1])
        row[i % width] = "*"
        blocks[height - level - 1] = "".join(row)

    print("\nASCII Equity Curve\n")
    print("\n".join(blocks))
    print(f"\nMin: ${min_val:,.0f}, Max: ${max_val:,.0f}")

ascii_plot(portfolio_history, width=80, height=20)


# -----------------------------
# MATPLOTLIB EQUITY CURVE
# -----------------------------
plt.figure(figsize=(12,6))
plt.plot(portfolio_history.index, portfolio_history.values, label='Portfolio', color='blue')
plt.title('Equity Curve - Thresholdless Momentum (2% per trade)')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
plt.show()