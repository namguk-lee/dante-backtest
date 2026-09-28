#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dante-style full-market backtest v3

Purpose
-------
Quantify a public-rule approximation of the '224 / bowl / anchor-volume' setup
across Korean equities without relying on proprietary indicators.

Primary base signal (A)
-----------------------
1) adjusted close crosses above EMA224
2) >= 60 of the prior 80 sessions closed below EMA224
3) EMA112 < EMA224 < EMA448
4) signal-day volume >= 1.5 x 20-day average volume
5) bullish signal candle
6) 40-session per-symbol signal cooldown

Stop-only comparison (profit target intentionally NOT used in v3)
-----------------------------------------------------------------
S1: intraday low <= signal EMA224 * 0.98
S2: close < current EMA224, exit next session open
S3: intraday low <= signal candle low * 0.995
S4: intraday low <= prior-20-session swing low * 0.995
S5: intraday low <= entry - 1.5 * ATR14(signal)

If no stop occurs, exit at the close after --horizon sessions.
This isolates the effect of the stop rule before target optimization.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd


ALIASES = {
    "series_id": ["series_id", "series", "episode_id"],
    "code": ["code", "ticker", "symbol"],
    "name": ["name"],
    "amount": ["amount", "turnover", "value"],
    "exchange": ["exchange", "market"],
    "date": ["date", "datetime", "time"],
    "open": ["open"],
    "high": ["high"],
    "low": ["low"],
    "close": ["close"],
    "adjusted_close": ["adjusted_close", "adj_close", "adj close", "adjclose", "adjusted close"],
    "volume": ["volume", "vol"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ko", type=Path, default=None)
    p.add_argument("--kq", type=Path, default=None)
    p.add_argument("--input", type=Path, action="append", default=[])
    p.add_argument("--out", type=Path, default=Path("dante_full_results"))
    p.add_argument("--signal-start", default="2018-01-01")
    p.add_argument("--horizon", type=int, default=60)
    p.add_argument("--cooldown", type=int, default=40)
    p.add_argument("--below-window", type=int, default=80)
    p.add_argument("--below-min", type=int, default=60)
    p.add_argument("--volume-multiple", type=float, default=1.5)
    p.add_argument("--min-bars", type=int, default=600)
    p.add_argument("--min-entry-price", type=float, default=0.0)
    p.add_argument("--min-avg-turnover-krw", type=float, default=0.0)
    p.add_argument("--roundtrip-cost-bps", type=float, default=0.0)
    p.add_argument("--train-end", default="2021-12-31")
    p.add_argument("--valid-end", default="2024-12-31")
    return p.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if ext in {".csv", ".txt"}:
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported file: {path}")


def normalize_columns(df: pd.DataFrame, default_exchange: str | None) -> pd.DataFrame:
    lower = {str(c).strip().lower(): c for c in df.columns}
    rename = {}
    for canonical, aliases in ALIASES.items():
        for a in aliases:
            if a.lower() in lower:
                rename[lower[a.lower()]] = canonical
                break
    df = df.rename(columns=rename).copy()

    required = ["code", "date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}; got {list(df.columns)}")
    if "adjusted_close" not in df.columns:
        df["adjusted_close"] = df["close"]
    if "exchange" not in df.columns:
        df["exchange"] = default_exchange or "UNKNOWN"
    if "name" not in df.columns:
        df["name"] = ""

    df["code"] = df["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    if "series_id" not in df.columns:
        df["series_id"] = df["exchange"].astype(str) + "_" + df["code"]
    df["exchange"] = df["exchange"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["open", "high", "low", "close", "adjusted_close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    df = df.dropna(subset=["code", "date", "open", "high", "low", "close", "adjusted_close", "volume"])
    df = df[(df["close"] > 0) & (df["adjusted_close"] > 0) & (df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0)]
    df = df.sort_values(["exchange", "series_id", "date"]).drop_duplicates(["exchange", "series_id", "date"], keep="last")
    return df


def load_inputs(args: argparse.Namespace) -> pd.DataFrame:
    frames = []
    paths = []
    if args.ko:
        paths.append((args.ko, "KO"))
    if args.kq:
        paths.append((args.kq, "KQ"))
    for p in args.input:
        paths.append((p, None))
    if not paths:
        raise SystemExit("Provide --ko/--kq or at least one --input file")
    for p, ex in paths:
        if not p.exists():
            raise FileNotFoundError(p)
        frames.append(normalize_columns(read_table(p), ex))
    return pd.concat(frames, ignore_index=True).sort_values(["exchange", "series_id", "date"])


def compute_indicators(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    factor = (g["adjusted_close"] / g["close"]).replace([np.inf, -np.inf], np.nan)
    factor = factor.ffill().bfill().fillna(1.0)
    g["aopen"] = g["open"] * factor
    g["ahigh"] = g["high"] * factor
    g["alow"] = g["low"] * factor
    g["aclose"] = g["adjusted_close"]

    for n in (112, 224, 448):
        g[f"ema{n}"] = g["aclose"].ewm(span=n, adjust=False, min_periods=n).mean()
    g["vol_ma20"] = g["volume"].rolling(20, min_periods=20).mean()
    if "amount" in g.columns and g["amount"].notna().any():
        amount = g["amount"].where(g["amount"] > 0, g["close"] * g["volume"])
        g["turnover20"] = amount.rolling(20, min_periods=20).mean()
    else:
        g["turnover20"] = (g["close"] * g["volume"]).rolling(20, min_periods=20).mean()

    prev_c = g["aclose"].shift(1)
    tr = pd.concat([
        g["ahigh"] - g["alow"],
        (g["ahigh"] - prev_c).abs(),
        (g["alow"] - prev_c).abs(),
    ], axis=1).max(axis=1)
    g["atr14"] = tr.rolling(14, min_periods=14).mean()
    g["daily_adj_ret"] = g["aclose"].pct_change()
    return g


def split_label(dt: pd.Timestamp, train_end: pd.Timestamp, valid_end: pd.Timestamp) -> str:
    if dt <= train_end:
        return "TRAIN"
    if dt <= valid_end:
        return "VALID"
    return "TEST"


def evaluate_fixed_stop(g: pd.DataFrame, entry_i: int, horizon: int, stop_level: float) -> tuple:
    entry = float(g.iloc[entry_i]["aopen"])
    end_i = min(entry_i + horizon, len(g) - 1)
    if not (math.isfinite(stop_level) and 0 < stop_level < entry):
        return float(g.iloc[end_i]["aclose"]), end_i, False, "TIME"
    for j in range(entry_i, end_i + 1):
        row = g.iloc[j]
        if float(row["aopen"]) <= stop_level:
            return float(row["aopen"]), j, True, "STOP_GAP"
        if float(row["alow"]) <= stop_level:
            return stop_level, j, True, "STOP"
    return float(g.iloc[end_i]["aclose"]), end_i, False, "TIME"


def evaluate_close_ema_stop(g: pd.DataFrame, entry_i: int, horizon: int) -> tuple:
    end_i = min(entry_i + horizon, len(g) - 1)
    for j in range(entry_i, end_i):
        c = float(g.iloc[j]["aclose"])
        e = float(g.iloc[j]["ema224"])
        if math.isfinite(e) and c < e:
            return float(g.iloc[j + 1]["aopen"]), j + 1, True, "EMA224_CLOSE"
    return float(g.iloc[end_i]["aclose"]), end_i, False, "TIME"


def scan_symbol(g: pd.DataFrame, args: argparse.Namespace, train_end: pd.Timestamp, valid_end: pd.Timestamp):
    g = compute_indicators(g).reset_index(drop=True)
    signals, trades = [], []
    n = len(g)
    if n < args.min_bars:
        return signals, trades, {
            "bars": n, "first_date": g["date"].min(), "last_date": g["date"].max(),
            "adj_jump_gt50pct": int((g["daily_adj_ret"].abs() > .50).sum()), "eligible": False,
        }

    signal_start = pd.Timestamp(args.signal_start)
    last_signal = -10_000
    start = max(448, args.below_window + 1)
    end = n - args.horizon - 1
    for i in range(start, end):
        if i - last_signal < args.cooldown:
            continue
        row, prev = g.iloc[i], g.iloc[i - 1]
        if pd.Timestamp(row["date"]) < signal_start:
            continue
        if not all(math.isfinite(float(row[c])) for c in ["ema112", "ema224", "ema448", "vol_ma20"]):
            continue

        if not (row["aclose"] > row["ema224"] and prev["aclose"] <= prev["ema224"]):
            continue
        hist = g.iloc[i - args.below_window:i]
        below_count = int((hist["aclose"] < hist["ema224"]).sum())
        if below_count < args.below_min:
            continue
        if not (row["ema112"] < row["ema224"] < row["ema448"]):
            continue
        if not (row["volume"] >= args.volume_multiple * row["vol_ma20"]):
            continue
        if not (row["aclose"] > row["aopen"]):
            continue
        if args.min_avg_turnover_krw > 0 and row["turnover20"] < args.min_avg_turnover_krw:
            continue

        entry_i = i + 1
        entry = float(g.iloc[entry_i]["aopen"])
        if entry < args.min_entry_price:
            continue

        last_signal = i
        sigdate = pd.Timestamp(row["date"])
        future60 = float(g.iloc[entry_i + args.horizon]["aclose"] / entry - 1)
        future20 = float(g.iloc[entry_i + min(20, args.horizon)]["aclose"] / entry - 1)
        swing_low = float(g.iloc[max(0, i - 19):i + 1]["alow"].min())
        atr14 = float(row["atr14"])

        signals.append({
            "exchange": row["exchange"], "series_id": row["series_id"], "code": row["code"], "name": row.get("name", ""),
            "signal_date": sigdate, "entry_date": g.iloc[entry_i]["date"], "entry": entry,
            "ema112": row["ema112"], "ema224": row["ema224"], "ema448": row["ema448"],
            "below_80": below_count, "volume_ratio": row["volume"] / row["vol_ma20"],
            "turnover20": row["turnover20"], "fwd20": future20, "fwd60": future60,
            "split": split_label(sigdate, train_end, valid_end),
        })

        methods = {
            "S1_EMA224_INTRADAY_2PCT": float(row["ema224"] * .98),
            "S3_SIGNAL_LOW": float(row["alow"] * .995),
            "S4_SWING20_LOW": float(swing_low * .995),
            "S5_ATR1_5": float(entry - 1.5 * atr14) if math.isfinite(atr14) else np.nan,
        }
        results = {name: evaluate_fixed_stop(g, entry_i, args.horizon, level) for name, level in methods.items()}
        results["S2_EMA224_CLOSE"] = evaluate_close_ema_stop(g, entry_i, args.horizon)

        for method, (exit_px, exit_i, stopped, reason) in results.items():
            gross = exit_px / entry - 1
            trades.append({
                "exchange": row["exchange"], "series_id": row["series_id"], "code": row["code"], "name": row.get("name", ""),
                "signal_date": sigdate, "entry_date": g.iloc[entry_i]["date"], "exit_date": g.iloc[exit_i]["date"],
                "method": method, "entry": entry, "exit": exit_px,
                "gross_return": gross, "net_return": gross - args.roundtrip_cost_bps / 10000.0,
                "hold_days": exit_i - entry_i, "stopped": stopped, "exit_reason": reason,
                "split": split_label(sigdate, train_end, valid_end),
            })

    dq = {
        "bars": n, "first_date": g["date"].min(), "last_date": g["date"].max(),
        "adj_jump_gt50pct": int((g["daily_adj_ret"].abs() > .50).sum()), "eligible": True,
    }
    return signals, trades, dq


def trimmed_mean(s: pd.Series, proportion: float = .10) -> float:
    x = np.sort(pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float))
    if len(x) == 0:
        return np.nan
    k = int(len(x) * proportion)
    if 2 * k >= len(x):
        return float(np.mean(x))
    return float(np.mean(x[k:len(x)-k]))


def summarize(trades: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    grouper = group_cols[0] if len(group_cols) == 1 else group_cols
    for keys, q in trades.groupby(grouper, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = dict(zip(group_cols, keys))
        r = q["net_return"]
        rec.update({
            "n": len(q),
            "mean_return": r.mean(),
            "median_return": r.median(),
            "trim10_mean_return": trimmed_mean(r),
            "win_rate": (r > 0).mean(),
            "stop_rate": q["stopped"].mean(),
            "avg_hold_days": q["hold_days"].mean(),
            "p10": r.quantile(.10),
            "p90": r.quantile(.90),
        })
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = load_inputs(args)
    train_end, valid_end = pd.Timestamp(args.train_end), pd.Timestamp(args.valid_end)

    all_signals, all_trades, quality = [], [], []
    groups = list(df.groupby(["exchange", "series_id"], sort=False))
    total = len(groups)
    for idx, ((ex, series_id), g) in enumerate(groups, 1):
        code = str(g.iloc[0]["code"])
        sigs, trs, dq = scan_symbol(g, args, train_end, valid_end)
        all_signals.extend(sigs)
        all_trades.extend(trs)
        quality.append({"exchange": ex, "series_id": series_id, "code": code, **dq})
        if idx % 100 == 0 or idx == total:
            print(f"[{idx}/{total}] signals={len(all_signals):,} trades={len(all_trades):,}", flush=True)

    signals = pd.DataFrame(all_signals)
    trades = pd.DataFrame(all_trades)
    quality_df = pd.DataFrame(quality)

    signals.to_csv(args.out / "signals.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(args.out / "stop_trades.csv", index=False, encoding="utf-8-sig")
    quality_df.to_csv(args.out / "data_quality.csv", index=False, encoding="utf-8-sig")

    if not trades.empty:
        trades["year"] = pd.to_datetime(trades["signal_date"]).dt.year
        summarize(trades, ["method"]).to_csv(args.out / "stop_comparison.csv", index=False, encoding="utf-8-sig")
        summarize(trades, ["split", "method"]).to_csv(args.out / "split_comparison.csv", index=False, encoding="utf-8-sig")
        summarize(trades, ["year", "method"]).to_csv(args.out / "yearly_comparison.csv", index=False, encoding="utf-8-sig")
        summarize(trades, ["exchange", "method"]).to_csv(args.out / "exchange_comparison.csv", index=False, encoding="utf-8-sig")

        liq_rows = []
        if not signals.empty:
            sig_liq = signals[["series_id", "signal_date", "turnover20"]].copy()
            tmp = trades.merge(sig_liq, on=["series_id", "signal_date"], how="left")
            for threshold in [0, 500_000_000, 1_000_000_000, 5_000_000_000]:
                q = tmp[tmp["turnover20"] >= threshold].copy()
                if q.empty:
                    continue
                z = summarize(q, ["split", "method"])
                z["min_turnover20"] = threshold
                liq_rows.append(z)
        if liq_rows:
            pd.concat(liq_rows, ignore_index=True).to_csv(args.out / "liquidity_comparison.csv", index=False, encoding="utf-8-sig")

        cost_rows = []
        for bps in [0, 20, 40]:
            q = trades.copy()
            q["net_return"] = q["gross_return"] - bps / 10000.0
            z = summarize(q, ["split", "method"])
            z["roundtrip_cost_bps"] = bps
            cost_rows.append(z)
        pd.concat(cost_rows, ignore_index=True).to_csv(args.out / "cost_sensitivity.csv", index=False, encoding="utf-8-sig")

    manifest = f"""Dante full-market backtest v3
rows={len(df):,}
series={df[['exchange','series_id']].drop_duplicates().shape[0]:,}
signal_start={args.signal_start}
horizon={args.horizon}
cooldown={args.cooldown}
below_window={args.below_window}
below_min={args.below_min}
volume_multiple={args.volume_multiple}
roundtrip_cost_bps={args.roundtrip_cost_bps}
train_end={args.train_end}
valid_end={args.valid_end}
NOTE: Signal rules are a quantitative research approximation, not an official proprietary indicator formula.
"""
    (args.out / "run_manifest.txt").write_text(manifest, encoding="utf-8")
    print(f"Done -> {args.out.resolve()}")


if __name__ == "__main__":
    main()
