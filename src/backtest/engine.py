"""
engine.py

Simple walk-forward backtesting engine for S&P 100 stocks.
Uses features computed in features.py to generate signals.

Strategy: Momentum-based
- Buy if Momentum_5 > 0
- Sell/No position if Momentum_5 <= 0
- Equal-weight portfolio
"""

import os
import pandas as pd
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------------
# CONFIGURATION
# -----------------------------
PROCESSED_DIR = "../../data/processed"
LOG_DIR = "../../logs"
os.makedirs(LOG_DIR, exist_ok=True)

INITIAL_CAPITAL = 100000  # starting portfolio value
POSITION_SIZE = 1.0        # fraction of portfolio per stock (equal weight)
LOOKBACK_FEATURE = 'Momentum_5'

# -----------------------------
# SETUP LOGGING
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
        logging.warning(f"No processed data for {ticker}")
        return pd.DataFrame()
    
    dfs = []
    for year_folder in os.listdir(ticker_dir):
        file_path = os.path.join(ticker_dir, year_folder, "features.parquet")
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            df['Ticker'] = ticker
            dfs.append(df)
    if dfs:
        return pd.concat(dfs).sort_values("Date")
    return pd.DataFrame()

# -----------------------------
# LOAD ALL TICKERS
# -----------------------------
tickers = [d.split("=")[1] for d in os.listdir(PROCESSED_DIR) if d.startswith("ticker=")]
all_data = []

for ticker in tqdm(tickers, desc="Loading features"):
    df = load_features(ticker)
    if not df.empty:
        all_data.append(df)

if not all_data:
    logging.error("No data found. Exiting.")
    exit()

data = pd.concat(all_data).sort_values("Date")
logging.info(f"Loaded features for {len(tickers)} tickers.")

# -----------------------------
# BACKTEST LOGIC
# -----------------------------
# Pivot to have tickers as columns
returns_pivot = data.pivot(index='Date', columns='Ticker', values='Return')
signal_pivot = data.pivot(index='Date', columns='Ticker', values=LOOKBACK_FEATURE)

# Generate simple momentum signals: 1 = buy, 0 = no position
signals = (signal_pivot > 0).astype(int)

# Calculate daily portfolio returns
weights = signals.div(signals.sum(axis=1), axis=0).fillna(0)  # equal weight among active positions
portfolio_returns = (weights * returns_pivot).sum(axis=1)

# Calculate cumulative portfolio value
portfolio_value = (1 + portfolio_returns).cumprod() * INITIAL_CAPITAL

# -----------------------------
# PERFORMANCE METRICS
# -----------------------------
def sharpe_ratio(returns, risk_free=0.0):
    return (returns.mean() - risk_free) / returns.std() * (252 ** 0.5)  # annualized

def max_drawdown(portfolio):
    cum_max = portfolio.cummax()
    drawdown = (portfolio - cum_max) / cum_max
    return drawdown.min()

sharpe = sharpe_ratio(portfolio_returns)
drawdown = max_drawdown(portfolio_value)
cagr = (portfolio_value.iloc[-1] / INITIAL_CAPITAL) ** (1 / (len(portfolio_value)/252)) - 1

logging.info(f"CAGR: {cagr:.2%}")
logging.info(f"Sharpe Ratio: {sharpe:.2f}")
logging.info(f"Max Drawdown: {drawdown:.2%}")

# -----------------------------
# PLOT PORTFOLIO
# -----------------------------
plt.figure(figsize=(10,6))
plt.plot(portfolio_value, label='Momentum Portfolio')
plt.title('Portfolio Equity Curve')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.grid(True)
plt.show()

logging.info("Backtest complete!")
