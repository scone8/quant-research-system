import os

from relative_strength_strategy import StrategyConfig, compute_signals, load_all_features


PROCESSED_DIR = "../../data/processed"
OUTPUT_DIR = "../../data/processed/strategy_signals"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "tomorrow_signals.csv")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = load_all_features(PROCESSED_DIR)
    if data.empty:
        raise RuntimeError("No processed features found. Run data ingestion/feature scripts first.")

    config = StrategyConfig(
        top_percentile=0.80,  # top 20%
        sma_short=20,
        sma_long=200,
        breakout_window=20,
        pullback_lookback=10,
        hold_days=5,
    )

    signals, _ = compute_signals(data, config=config)
    if signals.empty:
        raise RuntimeError("Signal generation returned no rows.")

    latest_date = signals["Date"].max()
    latest = signals[(signals["Date"] == latest_date) & (signals["Ticker"] != "SPY")].copy()

    out_cols = [
        "Date",
        "Ticker",
        "Close",
        "RS_Score",
        "RS_Percentile",
        "Eligible",
        "Buy_Pullback",
        "Exit_Pullback",
        "Buy_Breakout",
        "Exit_Breakout",
    ]
    latest = latest[out_cols].sort_values("Ticker")
    latest.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved latest-day signals to {OUTPUT_FILE}")
    print(latest.to_string(index=False))


if __name__ == "__main__":
    main()
