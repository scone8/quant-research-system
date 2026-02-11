# Systematic Equity Research Pipeline

## Overview
This project implements an end-to-end quantitative research pipeline for systematic equity trading using **S&P 100 stocks**. 

The goal is to simulate a production-grade research environment, focusing on **data engineering, reproducibility, and realistic strategy evaluation**. This project is designed to demonstrate strong data engineering and quantitative analysis skills for career purposes, particularly in **data engineering or quantitative finance**.

## Features
- **Automated Historical Data Ingestion:** Pulls OHLCV data for S&P 100 equities over the last 15+ years.
- **Structured Data Storage:** Partitioned Parquet files to simulate a scalable data lake.
- **Feature Engineering Pipeline:** Computes returns, momentum indicators, volatility, and volume-based features.
- **Walk-Forward Backtesting:** Evaluates strategies on unseen future data to prevent data leakage.
- **Risk and Performance Metrics:** Sharpe ratio, maximum drawdown, CAGR, and win rate.
- **Machine Learning Signals:** Baseline logistic regression and XGBoost classifiers for predicting positive next-day returns.
- **Dashboard (Future):** Streamlit-based interface for strategy and portfolio visualization.

## Project Scope
- **Universe:** S&P 100 stocks  
- **Frequency:** Daily OHLCV data  
- **Storage:** Partitioned Parquet files (S3-ready)  
- **Evaluation:** Walk-forward validation including transaction costs  
- **Position Sizing:** Volatility-adjusted allocation (planned for risk module)

## Architecture
Data Ingestion → Feature Engineering → Model Training → Backtesting → Performance Evaluation → Dashboard

- **Data Ingestion:** Downloads, cleans, and stores raw historical data.
- **Feature Engineering:** Calculates technical indicators and statistical features.
- **Model Training:** Trains ML classifiers to predict probability of positive returns.
- **Backtesting:** Evaluates strategies using realistic walk-forward simulation.
- **Dashboard:** Visualizes results, equity curves, and metrics.

## Folder Structure
quant-research-system/
├── data/                 # Raw and processed data
├── notebooks/            # Exploratory analysis and experiments
├── src/                  # Source code
│   ├── data/             # Data ingestion and processing
│   ├── models/           # ML training and prediction
│   ├── backtest/         # Backtesting engine and metrics
│   └── utils/            # Logging, helper functions
├── tests/                # Unit and integration tests
├── requirements.txt
├── README.md
└── Dockerfile

## Installation
1. Clone the repo:
git clone https://github.com/yourusername/quant-research-system.git
cd quant-research-system

2. Create virtual environment:
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows

3. Install dependencies:
pip install -r requirements.txt

## Getting Started
1. Populate `data/universe/sp100.csv` with the S&P 100 tickers.
2. Run the ingestion script:
python src/data/ingest.py
3. Features will be processed into `data/processed/`.
4. Backtesting and ML modules can then be executed using `src/models/train.py` and `src/backtest/engine.py`.

## Contributing
This project is structured for personal portfolio purposes. Contributions are welcome in the form of:
- Adding new features or indicators
- Improving backtesting logic
- Enhancing dashboard visualization

Please follow clean coding standards and document all changes.

## License
This project is for educational and portfolio purposes. No financial advice is offered. Use at your own risk.
