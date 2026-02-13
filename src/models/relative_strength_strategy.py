"""
relative_strength_strategy.py

Relative-strength stock selection with trend filter and two entry variants:
- Pullback entry
- Breakout entry

Outputs buy/exit signals and simple daily equity curves for comparison.
"""

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd


@dataclass(frozen=True)
class StrategyConfig:
    mom_20: int = 20
    mom_60: int = 60
    mom_120: int = 120
    mom_252: int = 252
    vol_window: int = 60
    top_percentile: float = 0.80
    sma_short: int = 20
    sma_long: int = 200
    breakout_window: int = 20
    pullback_lookback: int = 10
    hold_days: int = 5
    vol_floor: float = 1e-6


def _load_features_for_ticker(processed_dir: str, ticker: str) -> pd.DataFrame:
    ticker_dir = os.path.join(processed_dir, f"ticker={ticker}")
    if not os.path.exists(ticker_dir):
        return pd.DataFrame()

    frames = []
    for year_folder in os.listdir(ticker_dir):
        file_path = os.path.join(ticker_dir, year_folder, "features.parquet")
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            df["Ticker"] = ticker
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["Date"] = pd.to_datetime(out["Date"])
    out = out.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    return out


def load_all_features(processed_dir: str) -> pd.DataFrame:
    tickers = [d.split("=")[1] for d in os.listdir(processed_dir) if d.startswith("ticker=")]
    all_frames = []

    for ticker in tickers:
        df = _load_features_for_ticker(processed_dir, ticker)
        if not df.empty and "Close" in df.columns:
            all_frames.append(df[["Date", "Ticker", "Close"]].copy())

    if not all_frames:
        return pd.DataFrame(columns=["Date", "Ticker", "Close"])

    data = pd.concat(all_frames, ignore_index=True)
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data = data.dropna(subset=["Date", "Ticker", "Close"])
    data = data.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    return data


def _build_benchmark(data: pd.DataFrame) -> pd.DataFrame:
    # Prefer SPY if present, otherwise use equal-weight universe proxy.
    if "SPY" in data["Ticker"].unique():
        bench = (
            data[data["Ticker"] == "SPY"][["Date", "Close"]]
            .drop_duplicates(subset=["Date"], keep="last")
            .rename(columns={"Close": "Bench_Close"})
            .sort_values("Date")
        )
        bench["Bench_Source"] = "SPY"
    else:
        bench = (
            data.groupby("Date", as_index=False)["Close"]
            .mean()
            .rename(columns={"Close": "Bench_Close"})
            .sort_values("Date")
        )
        bench["Bench_Source"] = "EqualWeightUniverseProxy"

    return bench


def _apply_trading_state(df: pd.DataFrame, buy_col: str, prefix: str, hold_days: int) -> pd.DataFrame:
    buy_signal_col = f"Buy_{prefix}"
    exit_signal_col = f"Exit_{prefix}"
    position_col = f"Position_{prefix}"

    buy_signal = []
    exit_signal = []
    position = []
    in_position = False
    days_held = 0

    for _, row in df.iterrows():
        do_buy = False
        do_exit = False

        if in_position:
            days_held += 1

        if in_position and days_held >= hold_days:
            do_exit = True
            in_position = False
            days_held = 0
        elif (not in_position) and bool(row[buy_col]):
            do_buy = True
            in_position = True
            days_held = 0

        buy_signal.append(int(do_buy))
        exit_signal.append(int(do_exit))
        position.append(int(in_position))

    df[buy_signal_col] = buy_signal
    df[exit_signal_col] = exit_signal
    df[position_col] = position
    return df


def compute_signals(data: pd.DataFrame, config: StrategyConfig = StrategyConfig()) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()

    bench = _build_benchmark(data)
    bench = bench.sort_values("Date")
    bench["Bench_Return"] = bench["Bench_Close"].pct_change().fillna(0)
    bench["Bench_SMA_200"] = bench["Bench_Close"].rolling(config.sma_long).mean()
    bench["Market_Regime_On"] = bench["Bench_Close"] > bench["Bench_SMA_200"]

    bench["Bench_Mom20"] = bench["Bench_Close"].pct_change(config.mom_20)
    bench["Bench_Mom60"] = bench["Bench_Close"].pct_change(config.mom_60)
    bench["Bench_Mom120"] = bench["Bench_Close"].pct_change(config.mom_120)
    bench["Bench_Mom252"] = bench["Bench_Close"].pct_change(config.mom_252)
    bench["Bench_Mom_Composite"] = (
        0.20 * bench["Bench_Mom20"]
        + 0.40 * bench["Bench_Mom60"]
        + 0.25 * bench["Bench_Mom120"]
        + 0.15 * bench["Bench_Mom252"]
    )
    bench["Bench_Vol"] = bench["Bench_Return"].rolling(config.vol_window).std() * (252 ** 0.5)
    bench["Bench_VolAdj"] = bench["Bench_Mom_Composite"] / bench["Bench_Vol"].clip(lower=config.vol_floor)

    work = data.merge(
        bench[["Date", "Bench_Close", "Bench_VolAdj", "Market_Regime_On", "Bench_Source"]],
        on="Date",
        how="left",
    )
    work = work.sort_values(["Ticker", "Date"]).copy()
    work["Is_Benchmark_Ticker"] = work["Ticker"].eq("SPY")

    g = work.groupby("Ticker", group_keys=False)
    work["Return"] = g["Close"].pct_change().fillna(0)
    work["SMA_20"] = g["Close"].transform(lambda s: s.rolling(config.sma_short).mean())
    work["SMA_200"] = g["Close"].transform(lambda s: s.rolling(config.sma_long).mean())
    work["Mom20"] = g["Close"].pct_change(config.mom_20)
    work["Mom60"] = g["Close"].pct_change(config.mom_60)
    work["Mom120"] = g["Close"].pct_change(config.mom_120)
    work["Mom252"] = g["Close"].pct_change(config.mom_252)
    work["Mom_Composite"] = (
        0.20 * work["Mom20"]
        + 0.40 * work["Mom60"]
        + 0.25 * work["Mom120"]
        + 0.15 * work["Mom252"]
    )
    work["Vol_60"] = g["Return"].transform(lambda s: s.rolling(config.vol_window).std()) * (252 ** 0.5)
    work["Vol_Adj_Mom"] = work["Mom_Composite"] / work["Vol_60"].clip(lower=config.vol_floor)
    work["RS_Score"] = work["Vol_Adj_Mom"] - work["Bench_VolAdj"]
    work["RS_Percentile"] = pd.NA
    tradable_mask = ~work["Is_Benchmark_Ticker"]
    work.loc[tradable_mask, "RS_Percentile"] = (
        work.loc[tradable_mask].groupby("Date")["RS_Score"].rank(method="average", pct=True)
    )

    work["Eligible"] = (
        (work["RS_Percentile"] >= config.top_percentile)
        & (work["Close"] > work["SMA_200"])
        & (work["Market_Regime_On"].fillna(False))
        & tradable_mask
    )

    below_20 = (work["Close"] < work["SMA_20"]).astype(int)
    work["Had_Pullback"] = below_20.groupby(work["Ticker"]).transform(
        lambda s: s.rolling(config.pullback_lookback, min_periods=1).max().shift(1).fillna(0).astype(bool)
    )
    work["BuyRaw_Pullback"] = work["Eligible"] & work["Had_Pullback"] & (work["Close"] > work["SMA_20"])

    work["Prior_High"] = g["Close"].transform(
        lambda s: s.shift(1).rolling(config.breakout_window, min_periods=config.breakout_window).max()
    )
    work["BuyRaw_Breakout"] = work["Eligible"] & (work["Close"] > work["Prior_High"])

    per_ticker = []
    for _, ticker_df in work.groupby("Ticker", sort=False):
        ticker_df = ticker_df.sort_values("Date").copy()
        ticker_df = _apply_trading_state(
            ticker_df,
            "BuyRaw_Pullback",
            "Pullback",
            hold_days=config.hold_days,
        )
        ticker_df = _apply_trading_state(
            ticker_df,
            "BuyRaw_Breakout",
            "Breakout",
            hold_days=config.hold_days,
        )
        per_ticker.append(ticker_df)

    signals = pd.concat(per_ticker, ignore_index=True).sort_values(["Date", "Ticker"]).reset_index(drop=True)
    return signals, bench


def build_backtest_curves(
    signals: pd.DataFrame,
    bench: pd.DataFrame,
    backtest_start: str = "2023-01-01",
    initial_capital: float = 1000.0,
) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()

    signals = signals.sort_values(["Ticker", "Date"]).copy()
    signals["PrevPos_Pullback"] = signals.groupby("Ticker")["Position_Pullback"].shift(1).fillna(0)
    signals["PrevPos_Breakout"] = signals.groupby("Ticker")["Position_Breakout"].shift(1).fillna(0)

    daily = signals.groupby("Date").apply(
        lambda d: pd.Series(
            {
                "Ret_Pullback": (
                    (d["PrevPos_Pullback"] * d["Return"]).sum() / d["PrevPos_Pullback"].sum()
                    if d["PrevPos_Pullback"].sum() > 0
                    else 0.0
                ),
                "Ret_Breakout": (
                    (d["PrevPos_Breakout"] * d["Return"]).sum() / d["PrevPos_Breakout"].sum()
                    if d["PrevPos_Breakout"].sum() > 0
                    else 0.0
                ),
            }
        )
    )
    daily = daily.reset_index()

    bench_daily = bench[["Date", "Bench_Return"]].copy()
    curves = daily.merge(bench_daily, on="Date", how="left").fillna(0)
    curves = curves.sort_values("Date")
    start_dt = pd.to_datetime(backtest_start)
    curves = curves[curves["Date"] >= start_dt].copy()
    if curves.empty:
        return curves

    curves["Equity_Pullback"] = (1 + curves["Ret_Pullback"]).cumprod()
    curves["Equity_Breakout"] = (1 + curves["Ret_Breakout"]).cumprod()
    curves["Equity_Benchmark"] = (1 + curves["Bench_Return"]).cumprod()
    curves["Equity_Pullback_USD"] = initial_capital * curves["Equity_Pullback"]
    curves["Equity_Breakout_USD"] = initial_capital * curves["Equity_Breakout"]
    curves["Equity_Benchmark_USD"] = initial_capital * curves["Equity_Benchmark"]
    return curves


def plot_curves(curves: pd.DataFrame, output_path: str, initial_capital: float = 1000.0, backtest_start: str = "2023-01-01") -> None:
    if curves.empty:
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(curves["Date"], curves["Equity_Pullback_USD"], label="Pullback", linewidth=1.8)
    plt.plot(curves["Date"], curves["Equity_Breakout_USD"], label="Breakout", linewidth=1.8)
    plt.plot(curves["Date"], curves["Equity_Benchmark_USD"], label="Benchmark", linewidth=1.3, alpha=0.8)
    plt.title(f"Relative Strength + Trend Filter ({backtest_start}+)")
    plt.xlabel("Date")
    plt.ylabel(f"Portfolio Value ($, Start = {initial_capital:,.0f})")
    plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def compute_metrics(curves: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    if curves.empty:
        return {}

    def _sharpe(ret: pd.Series) -> float:
        vol = ret.std()
        if vol == 0 or pd.isna(vol):
            return 0.0
        return float((ret.mean() / vol) * (252 ** 0.5))

    results = {}
    mapping = {
        "Pullback": ("Ret_Pullback", "Equity_Pullback_USD"),
        "Breakout": ("Ret_Breakout", "Equity_Breakout_USD"),
        "Benchmark": ("Bench_Return", "Equity_Benchmark_USD"),
    }
    for name, (ret_col, equity_col) in mapping.items():
        ret = curves[ret_col].fillna(0)
        eq = curves[equity_col].fillna(method="ffill")
        results[name] = {
            "Sharpe": _sharpe(ret),
            "FinalCapital": float(eq.iloc[-1]),
        }
    return results
