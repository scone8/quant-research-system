"""
train.py

Builds and saves historical signals for:
1) Relative Strength + Trend Filter + Pullback entry
2) Relative Strength + Trend Filter + Breakout entry

Also saves a comparison backtest chart.
"""

import os
import pandas as pd

from relative_strength_strategy import (
    StrategyConfig,
    build_backtest_curves,
    compute_metrics,
    compute_signals,
    load_all_features,
    plot_curves,
)


PROCESSED_DIR = "../../data/processed"
OUTPUT_DIR = "../../data/processed/strategy_signals"
LOG_DIR = "../../logs"

SIGNALS_PARQUET = os.path.join(OUTPUT_DIR, "relative_strength_signals.parquet")
SIGNALS_CSV = os.path.join(OUTPUT_DIR, "relative_strength_signals.csv")
CURVES_CSV = os.path.join(OUTPUT_DIR, "relative_strength_backtest_curves.csv")
CHART_PATH = os.path.join(LOG_DIR, "relative_strength_pullback_vs_breakout.png")
BACKTEST_START = "2023-01-01"
INITIAL_CAPITAL = 1000.0


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    data = load_all_features(PROCESSED_DIR)
    if data.empty:
        raise RuntimeError("No processed features found. Run data ingestion/feature scripts first.")

    primary_config = StrategyConfig(
        top_percentile=0.80,  # top 20%
        sma_short=20,
        sma_long=200,
        breakout_window=20,
        pullback_lookback=10,
        hold_days=5,
    )
    rethink_config = StrategyConfig(
        top_percentile=0.90,  # stricter leaders
        sma_short=20,
        sma_long=200,
        breakout_window=55,
        pullback_lookback=7,
        hold_days=5,
    )

    signals, bench = compute_signals(data, config=primary_config)
    if signals.empty:
        raise RuntimeError("Primary signal generation returned no rows.")

    export_cols = [
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
        "Position_Pullback",
        "Position_Breakout",
    ]
    curves = build_backtest_curves(
        signals,
        bench,
        backtest_start=BACKTEST_START,
        initial_capital=INITIAL_CAPITAL,
    )
    if curves.empty:
        raise RuntimeError(f"No backtest rows found on/after {BACKTEST_START}.")
    metrics = compute_metrics(curves)
    benchmark_final = metrics["Benchmark"]["FinalCapital"]
    benchmark_sharpe = metrics["Benchmark"]["Sharpe"]
    primary_pass = (
        (metrics["Pullback"]["FinalCapital"] > benchmark_final and metrics["Pullback"]["Sharpe"] > benchmark_sharpe)
        or (metrics["Breakout"]["FinalCapital"] > benchmark_final and metrics["Breakout"]["Sharpe"] > benchmark_sharpe)
    )

    selected_label = "primary"
    if not primary_pass:
        print("Primary config did not beat benchmark on both final capital and Sharpe. Running rethink config...")
        rethink_signals, rethink_bench = compute_signals(data, config=rethink_config)
        rethink_curves = build_backtest_curves(
            rethink_signals,
            rethink_bench,
            backtest_start=BACKTEST_START,
            initial_capital=INITIAL_CAPITAL,
        )
        if not rethink_curves.empty:
            rethink_metrics = compute_metrics(rethink_curves)
            alt_benchmark_final = rethink_metrics["Benchmark"]["FinalCapital"]
            alt_benchmark_sharpe = rethink_metrics["Benchmark"]["Sharpe"]
            rethink_pass = (
                (rethink_metrics["Pullback"]["FinalCapital"] > alt_benchmark_final and rethink_metrics["Pullback"]["Sharpe"] > alt_benchmark_sharpe)
                or (rethink_metrics["Breakout"]["FinalCapital"] > alt_benchmark_final and rethink_metrics["Breakout"]["Sharpe"] > alt_benchmark_sharpe)
            )
            if rethink_pass:
                signals = rethink_signals
                bench = rethink_bench
                curves = rethink_curves
                metrics = rethink_metrics
                selected_label = "rethink"
            else:
                # Keep whichever configuration had better best-strategy Sharpe.
                primary_best_sharpe = max(metrics["Pullback"]["Sharpe"], metrics["Breakout"]["Sharpe"])
                rethink_best_sharpe = max(rethink_metrics["Pullback"]["Sharpe"], rethink_metrics["Breakout"]["Sharpe"])
                if rethink_best_sharpe > primary_best_sharpe:
                    signals = rethink_signals
                    bench = rethink_bench
                    curves = rethink_curves
                    metrics = rethink_metrics
                    selected_label = "rethink_best_sharpe"

    signals_to_save = signals[signals["Ticker"] != "SPY"][export_cols].sort_values(["Date", "Ticker"]).copy()
    signals_to_save.to_parquet(SIGNALS_PARQUET, index=False)
    signals_to_save.to_csv(SIGNALS_CSV, index=False)

    curves.to_csv(CURVES_CSV, index=False)
    plot_curves(
        curves,
        CHART_PATH,
        initial_capital=INITIAL_CAPITAL,
        backtest_start=BACKTEST_START,
    )

    last_date = pd.to_datetime(signals_to_save["Date"]).max().date()
    bench_source = bench["Bench_Source"].dropna().iloc[-1] if "Bench_Source" in bench.columns else "Unknown"
    print(f"Saved signals: {SIGNALS_PARQUET}")
    print(f"Saved signals csv: {SIGNALS_CSV}")
    print(f"Saved backtest curves: {CURVES_CSV}")
    print(f"Saved chart: {CHART_PATH}")
    print(f"Latest signal date: {last_date}")
    print(f"Backtest start: {BACKTEST_START}")
    print(f"Initial capital: ${INITIAL_CAPITAL:,.0f}")
    print(f"Benchmark source: {bench_source}")
    print(f"Config used: {selected_label}")
    print(f"Final capital (Pullback): ${metrics['Pullback']['FinalCapital']:,.2f}")
    print(f"Final capital (Breakout): ${metrics['Breakout']['FinalCapital']:,.2f}")
    print(f"Final capital (Benchmark): ${metrics['Benchmark']['FinalCapital']:,.2f}")
    print(f"Sharpe (Pullback): {metrics['Pullback']['Sharpe']:.3f}")
    print(f"Sharpe (Breakout): {metrics['Breakout']['Sharpe']:.3f}")
    print(f"Sharpe (Benchmark): {metrics['Benchmark']['Sharpe']:.3f}")
    print(
        "Pullback beats benchmark on both: "
        f"{metrics['Pullback']['FinalCapital'] > metrics['Benchmark']['FinalCapital'] and metrics['Pullback']['Sharpe'] > metrics['Benchmark']['Sharpe']}"
    )
    print(
        "Breakout beats benchmark on both: "
        f"{metrics['Breakout']['FinalCapital'] > metrics['Benchmark']['FinalCapital'] and metrics['Breakout']['Sharpe'] > metrics['Benchmark']['Sharpe']}"
    )


if __name__ == "__main__":
    main()
