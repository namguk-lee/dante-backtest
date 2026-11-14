#!/usr/bin/env python3
"""Daily as-of Dante base-length sensitivity study (research only).

All signals use information available by signal close. Entries are at the next
session's adjusted open; stops/targets use next-session OHLC with stop-first
ambiguity and gap-aware fills. These are PUBLIC-rule approximations, not
proprietary Dante signals. Cross-sectional marcap input retains delistings.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

LENS = (40, 60, 90)
LIQUIDITY = (1_000_000_000, 5_000_000_000)
HORIZONS = (20, 60)
COST = 0.004
COOLDOWN = 40
PROFILES = [(n, amt) for n in LENS for amt in LIQUIDITY]


def base_ok(p, start, end, cap=0.35):
    """An actual completed pre-signal base, excluding the event candle."""
    v = p[start:end]
    if len(v) < 40 or not np.isfinite(v).all() or np.min(v) <= 0:
        return False, np.nan, np.nan
    q10, q90 = np.quantile(v, [0.1, 0.9])
    spread = float(q90 / q10 - 1.0)
    change = float(v[-1] / v[0] - 1.0)
    m = max(10, len(v) // 3)
    delayed_low = float(np.min(v[m:])) < float(np.min(v[:m])) * 0.92
    drawdown = float(np.min(v / np.maximum.accumulate(v) - 1.0))
    if spread > cap or change < -0.20 or change > 0.22 or delayed_low or drawdown < -0.28:
        return False, spread, change
    return True, spread, change


def trade_path(o, hi, lo, c, signal_i, entry, stop, horizon):
    """Returns only mature outcomes; takes stop before target on ambiguous bars."""
    last = signal_i + horizon
    if last >= len(c):
        return None
    target = entry + 2.0 * (entry - stop)
    outcome = "TIMEOUT"
    price = float(c[last])
    held = horizon
    for j in range(signal_i + 1, last + 1):
        if not all(np.isfinite(q) and q > 0 for q in (o[j], hi[j], lo[j], c[j])):
            return None
        # Open gaps are executable before intraday stop/target checks.
        if o[j] <= stop:
            price = float(o[j]); outcome = "STOP_GAP"; held = j - signal_i; break
        if o[j] >= target:
            price = float(o[j]); outcome = "TARGET_GAP"; held = j - signal_i; break
        if lo[j] <= stop:
            price = float(stop); outcome = "STOP"; held = j - signal_i; break
        if hi[j] >= target:
            price = float(target); outcome = "TARGET"; held = j - signal_i; break
    gross = price / entry - 1
    net = (gross - COST) * 100.0
    r_multiple = (price - entry) / (entry - stop) - COST * entry / (entry - stop)
    hold_net = (c[last] / entry - 1 - COST) * 100.0
    return dict(net=net, r_multiple=r_multiple, hold_net=hold_net,
                outcome=outcome, held=held)


def classify_event(i, close, op, high, low, volume, v20, ema112, ema224,
                   prior20high, last224cross):
    """Choose a single highest-priority event for each day."""
    if not all(np.isfinite(x) and x > 0 for x in
               (close[i], op[i], high[i], low[i], ema112[i], ema224[i], v20[i])):
        return None
    d = close[i] / ema224[i] - 1.0
    body = close[i] / op[i] - 1.0
    vol = volume[i] / v20[i]
    crossed224 = i > 0 and close[i] > ema224[i] and close[i-1] <= ema224[i-1]
    if crossed224 and -0.005 <= d <= .085 and vol >= 1.25 and close[i] >= ema112[i]:
        return "EMA224_BREAKOUT"
    if (0 <= d <= .055 and 1 <= i-last224cross <= 14 and
            low[i] <= ema224[i] * 1.025 and close[i] >= op[i] and
            close[i] >= ema112[i] and vol <= 1.8):
        return "EMA224_PULLBACK"
    if (body >= .03 and vol >= 1.6 and
            np.isfinite(prior20high[i]) and close[i] > prior20high[i] and
            -.13 <= d <= .085 and close[i] >= ema112[i]):
        return "PRE224_ANCHOR"
    return None


def scan_one(g):
    g = g.sort_values("date").reset_index(drop=True)
    if len(g) < 520:
        return []
    date = pd.to_datetime(g.date)
    raw = g.close.to_numpy(float)
    adj = g.adjusted_close.to_numpy(float)
    fac = np.divide(adj, raw, out=np.ones(len(g)), where=(raw > 0))
    o = g.open.to_numpy(float) * fac
    hi = g.high.to_numpy(float) * fac
    lo = g.low.to_numpy(float) * fac
    vol = g.volume.to_numpy(float)
    amount = g.amount.to_numpy(float)
    ema112 = pd.Series(adj).ewm(span=112, adjust=False, min_periods=112).mean().to_numpy()
    ema224 = pd.Series(adj).ewm(span=224, adjust=False, min_periods=224).mean().to_numpy()
    ema448 = pd.Series(adj).ewm(span=448, adjust=False, min_periods=448).mean().to_numpy()
    v20 = pd.Series(vol).rolling(20).mean().to_numpy()
    amt20 = pd.Series(amount).rolling(20).mean().to_numpy()
    prev20high = pd.Series(hi).shift(1).rolling(20).max().to_numpy()
    below80 = pd.Series((adj < ema224).astype(float)).shift(1).rolling(80).sum().to_numpy()
    last224cross = -10000
    recent_last = {k: -10000 for k in PROFILES}
    records = []
    for i in range(520, len(g) - 1):
        if date.iat[i] < pd.Timestamp("2018-01-01"):
            if adj[i] > ema224[i] and adj[i-1] <= ema224[i-1]:
                last224cross = i
            continue
        ev = classify_event(i, adj, o, hi, lo, vol, v20, ema112,
                            ema224, prev20high, last224cross)
        if adj[i] > ema224[i] and adj[i-1] <= ema224[i-1]:
            last224cross = i
        if not ev or not np.isfinite(below80[i]) or below80[i] < 38:
            continue
        if not np.isfinite(amt20[i]) or amt20[i] < LIQUIDITY[0]:
            continue
        if not np.isfinite(ema448[i]):
            continue
        choices = []
        for w in LENS:
            s = i-w
            if s < 260:
                continue
            valid, spread, change = base_ok(adj, s, i, cap=.35)
            if not valid:
                continue
            # The peak must predate the entire base; decline >= 20%.
            prior_peak = float(np.max(hi[max(0, s-260):s]))
            early_median = float(np.median(adj[s:min(s+10, i)]))
            if early_median <= 0 or prior_peak / early_median < 1.25:
                continue
            choices.append((w, spread, change, prior_peak))
        if not choices:
            continue
        entry = float(o[i+1])
        if not np.isfinite(entry) or entry <= 0:
            continue
        # Structure-based stop: recent swing low or 112/224 support.
        line = ema224[i] if adj[i] >= ema224[i] else ema112[i]
        stop = max(float(np.min(lo[max(0, i-14):i+1]))*.995, float(line)*.965)
        risk = 1 - stop/entry
        if not (.025 <= risk <= .18):
            continue
        fwd = {n: trade_path(o, hi, lo, adj, i, entry, stop, n) for n in HORIZONS}
        for w, spread, change, peak in choices:
            for amt in LIQUIDITY:
                key = (w, amt)
                if amt20[i] < amt or i - recent_last[key] < COOLDOWN:
                    continue
                recent_last[key] = i
                item = dict(date=date.iat[i].strftime("%Y-%m-%d"),
                            year=int(date.iat[i].year),
                            code=str(g.code.iloc[i]).zfill(6),
                            name=str(g.name.iloc[i]),
                            exchange=str(g.exchange.iloc[i]),
                            setup=ev, base_days=w, liquidity_krw=int(amt),
                            profile=f"B{w}_T{int(amt/1e8)}EOK",
                            signal_close_raw=float(raw[i]),
                            entry_open_adjusted=entry,
                            stop_adjusted=stop, stop_risk_pct=100*risk,
                            base_range_pct=100*spread, base_change_pct=100*change,
                            prior_peak_drop_pct=100*(early_median/peak-1),
                            dist224_pct=100*(adj[i]/ema224[i]-1),
                            vol_ratio=float(vol[i]/v20[i]),
                            below224_of_last80=int(below80[i]))
                for n in HORIZONS:
                    path = fwd[n]
                    for attr in ("net", "hold_net", "r_multiple", "held", "outcome"):
                        item[f"{attr}_{n}d"] = path.get(attr) if path else None
                records.append(item)
    return records


def summarize(records, group_cols):
    if records.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in records.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        r = dict(zip(group_cols, keys))
        r.update(signals=int(len(group)), stocks=int(group.code.nunique()))
        for h in HORIZONS:
            s = pd.to_numeric(group[f"net_{h}d"], errors="coerce").dropna()
            hold = pd.to_numeric(group[f"hold_net_{h}d"], errors="coerce").dropna()
            r.update({f"mature_{h}d":len(s),
                      f"mean_net_{h}d_pct":float(s.mean()) if len(s) else None,
                      f"median_net_{h}d_pct":float(s.median()) if len(s) else None,
                      f"win_{h}d_pct":float(100*(s>0).mean()) if len(s) else None,
                      f"mean_hold_{h}d_pct":float(hold.mean()) if len(hold) else None,
                      f"stop_rate_{h}d_pct":float(100*group[f"outcome_{h}d"].astype(str).str.startswith("STOP").mean()) if len(s) else None})
        rows.append(r)
    return pd.DataFrame(rows)


def self_test():
    ok,_,_ = base_ok(np.full(100,100.)+np.sin(np.arange(100)/5),0,90)
    fail,_,_ = base_ok(np.linspace(100,65,100),0,90)
    assert ok and not fail, "base detector synthetic test"
    # Gap through stop executes at open, and stop wins ambiguous stop/target day.
    op=np.array([100., 85., 100., 100.])
    hi=np.array([100., 105., 105., 105.])
    lo=np.array([100., 80., 90., 90.])
    close=np.array([100., 90., 101., 101.])
    x=trade_path(op,hi,lo,close,0,100.,90.,2)
    assert x["outcome"]=="STOP_GAP" and round(x["net"],1)==-15.4


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--ko",type=Path,required=True)
    p.add_argument("--kq",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--min-year",type=int,default=2018)
    a=p.parse_args()
    self_test()
    a.out.mkdir(parents=True,exist_ok=True)
    from dante_classic_scanner import load
    market=pd.concat([load(a.ko,"KO"),load(a.kq,"KQ")],ignore_index=True)
    print("PANEL",len(market),"first",market.date.min(),"last",market.date.max(),flush=True)
    all_items=[]
    for n,(_,group) in enumerate(market.groupby("series_id",sort=False),1):
        all_items.extend(scan_one(group))
        if n%300==0:
            print("episodes",n,"events",len(all_items),flush=True)
    e=pd.DataFrame(all_items)
    if e.empty:
        raise RuntimeError("No events detected; verify thresholds and panel coverage")
    e["split"]=np.select([e.year<=2021,e.year<=2024],["TRAIN","VALID"],default="TEST")
    e.sort_values(["date","code","profile"]).to_csv(a.out/"signals.csv",index=False,encoding="utf-8-sig")
    summary=summarize(e,["profile","split"])
    yearly=summarize(e,["profile","year"])
    setup=summarize(e,["profile","setup","split"])
    summary.to_csv(a.out/"summary_splits.csv",index=False,encoding="utf-8-sig")
    yearly.to_csv(a.out/"summary_yearly.csv",index=False,encoding="utf-8-sig")
    setup.to_csv(a.out/"summary_setups.csv",index=False,encoding="utf-8-sig")
    # Show signals/year separately from truly mature trade sample sizes.
    yearly_count=(e.groupby(["profile","year"]).size().unstack(fill_value=0)
                    .reset_index())
    yearly_count.to_csv(a.out/"signal_counts.csv",index=False,encoding="utf-8-sig")
    print("=== COUNTS BY YEAR ===",flush=True)
    print(yearly_count.to_string(index=False),flush=True)
    print("=== SPLIT RESULTS ===",flush=True)
    print(summary.to_string(index=False),flush=True)
    print("=== SETUP RESULTS ===",flush=True)
    print(setup.to_string(index=False),flush=True)
    print("Research only: next-open entries, gap-aware stops, 2R target; 40bp cost, no liquidity/slippage fill simulation.",flush=True)


if __name__=="__main__":
    main()
