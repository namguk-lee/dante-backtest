#!/usr/bin/env python3
"""Compare 60/90-day Gongguri + EMA224/anchor signals under TWO liquidity gates.

RESEARCH ONLY. Reuses original V4 public-pattern base shape and 2R simulation;
the *only* selection difference between arms is its liquidity gate.
- Legacy: current-inclusive 20-session amount >= KRW 5 billion.
- Cap-aware: previous-20 amount >= size-specific minimum AND
  previous-day listed market capitalization turnover >= size-specific ratio.
Forward 20-day outcomes are compared only on MATURE, actually observable
simulated trades. No paid Dante code, no trade orders.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from dante_base_v4_daily_backtest import base_ok, trade_path, classify_event
from dante_classic_scanner import load
from cap_liquidity import attach_liquidity_features, describe_liquidity

BASES=(60,90)
GATES=("LEGACY_50E","CAP_RELATIVE")
COST_PCT=0.4
HORIZONS=(20,60)
COOLDOWN_TRADING_DAYS=40

def scan_episode(frame):
    g=frame.sort_values("date").reset_index(drop=True)
    if len(g)<520:return []
    date=pd.to_datetime(g.date)
    raw=g.close.to_numpy(float)
    adj=g.adjusted_close.to_numpy(float)
    factor=np.divide(adj,raw,out=np.ones(len(g)),where=raw>0)
    o=g.open.to_numpy(float)*factor
    hi=g.high.to_numpy(float)*factor
    lo=g.low.to_numpy(float)*factor
    vol=g.volume.to_numpy(float)
    amount=g.amount.to_numpy(float)
    ema112=pd.Series(adj).ewm(span=112,adjust=False,min_periods=112).mean().to_numpy()
    ema224=pd.Series(adj).ewm(span=224,adjust=False,min_periods=224).mean().to_numpy()
    ema448=pd.Series(adj).ewm(span=448,adjust=False,min_periods=448).mean().to_numpy()
    v20=pd.Series(vol).rolling(20,min_periods=20).mean().to_numpy()
    amt20=pd.Series(amount).rolling(20,min_periods=20).mean().to_numpy()
    prev20high=pd.Series(hi).shift(1).rolling(20).max().to_numpy()
    below80=pd.Series((adj<ema224).astype(float)).shift(1).rolling(80).sum().to_numpy()
    liq=attach_liquidity_features(g)
    legacy_pass=np.isfinite(amt20)&(amt20>=5e9)
    cap_pass=liq.adaptive_liquidity_pass.to_numpy(dtype=bool)
    last224cross=-10000
    last_signal={(w,gate):-10000 for w in BASES for gate in GATES}
    events=[]
    for i in range(520,len(g)-1):
        if date.iat[i]<pd.Timestamp("2018-01-01"):
            if adj[i]>ema224[i] and adj[i-1]<=ema224[i-1]:
                last224cross=i
            continue
        ev=classify_event(i,adj,o,hi,lo,vol,v20,ema112,ema224,prev20high,last224cross)
        if adj[i]>ema224[i] and adj[i-1]<=ema224[i-1]:
            last224cross=i
        if not ev or not (legacy_pass[i] or cap_pass[i]):
            continue
        if not np.isfinite(below80[i]) or below80[i]<38 or not np.isfinite(ema448[i]):
            continue
        valid_bases=[]
        for w in BASES:
            s=i-w
            if s<260:continue
            valid,spread,change=base_ok(adj,s,i,cap=.35)
            if not valid:continue
            prior_peak=float(np.max(hi[max(0,s-260):s]))
            early_med=float(np.median(adj[s:min(s+10,i)]))
            if early_med<=0 or prior_peak/early_med<1.25:
                continue
            valid_bases.append((w,spread,change,prior_peak,early_med))
        if not valid_bases:continue
        entry=float(o[i+1])
        if not np.isfinite(entry) or entry<=0:continue
        line=ema224[i] if adj[i]>=ema224[i] else ema112[i]
        stop=max(float(np.min(lo[max(0,i-14):i+1]))*.995,float(line)*.965)
        risk=1-stop/entry
        if not(.025<=risk<=.18):continue
        fwd={h:trade_path(o,hi,lo,adj,i,entry,stop,h) for h in HORIZONS}
        tech_key=(str(date.iat[i].date()),str(g.code.iloc[i]).zfill(6),ev)
        for w,spread,change,peak,early_med in valid_bases:
            for gate,passed in zip(GATES,(bool(legacy_pass[i]),bool(cap_pass[i]))):
                if not passed or i-last_signal[(w,gate)]<COOLDOWN_TRADING_DAYS:
                    continue
                last_signal[(w,gate)]=i
                rec=dict(
                    date=str(date.iat[i].date()),year=int(date.iat[i].year),
                    code=str(g.code.iloc[i]).zfill(6),
                    name=str(g.name.iloc[i]),exchange=str(g.exchange.iloc[i]),
                    series_id=str(g.series_id.iloc[i]),
                    gate=gate,base_days=w,
                    setup=ev,signal_price_raw=float(raw[i]),
                    technical_event_key=":".join(tech_key),
                    entry_open_adjusted=entry,stop_adjusted=stop,
                    stop_risk_pct=risk*100,
                    base_range_pct=spread*100,
                    base_change_pct=change*100,
                    prior_peak_drop_pct=100*(early_med/peak-1),
                    dist224_pct=100*(adj[i]/ema224[i]-1),
                    volume_ratio=float(vol[i]/v20[i]),
                    below224_last80=int(below80[i]),
                    old_liquidity_eligible=bool(legacy_pass[i]),
                    cap_liquidity_eligible=bool(cap_pass[i]),
                    **describe_liquidity(liq.iloc[i]),
                )
                for h in HORIZONS:
                    sim=fwd[h]
                    for k in ("net","hold_net","r_multiple","held","outcome"):
                        rec[f"{k}_{h}d"]=sim.get(k) if sim is not None else None
                events.append(rec)
    return events

def outcome_summary(events):
    rows=[]
    e=events.copy()
    e["split"]=np.select([e.year<=2021,e.year<=2024],
                         ["TRAIN_2018_21","VALID_2022_24"],default="TEST_2025_26")
    for (base,setup,split,gate),g in e.groupby(["base_days","setup","split","gate"],dropna=False):
        m=g[g.net_20d.notna()&g.outcome_20d.notna()]
        o=m.outcome_20d.astype(str)
        rows.append(dict(
            base_days=base,setup=setup,split=split,gate=gate,
            events=len(g),unique_stocks=g.code.nunique(),
            mature20=len(m),immature20=len(g)-len(m),
            target20=int(o.str.startswith("TARGET").sum()),
            stopped20=int(o.str.startswith("STOP").sum()),
            timeout20=int(o.eq("TIMEOUT").sum()),
            target20_pct=float(100*o.str.startswith("TARGET").mean()) if len(m) else np.nan,
            mean_net20_pct=float(m.net_20d.mean()) if len(m) else np.nan,
            median_net20_pct=float(m.net_20d.median()) if len(m) else np.nan,
            win20_pct=float(100*m.net_20d.gt(0).mean()) if len(m) else np.nan,
            mean_hold20_pct=float(m.hold_net_20d.mean()) if len(m) else np.nan,
            median_cap_billion=float(g.cap_before_event_won.median()/1e9),
            median_amt_pre20_billion=float(g.pre20_amount_won.median()/1e9),
        ))
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--ko",type=Path,required=True)
    p.add_argument("--kq",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args();args.out.mkdir(exist_ok=True,parents=True)
    panel=pd.concat([load(args.ko,"KO"),load(args.kq,"KQ")],ignore_index=True)
    out=[]
    for n,(sid,g) in enumerate(panel.groupby("series_id",sort=False),1):
        out.extend(scan_episode(g))
        if n%300==0:print("episodes",n,"bowl_events",len(out),flush=True)
    if not out:raise RuntimeError("No B60/B90 research events; check input panel")
    e=pd.DataFrame(out).sort_values(["date","code","base_days","gate"])
    e.to_csv(args.out/"all_b60_b90_cap_vs_50e_simulated.csv",index=False,encoding="utf-8-sig")
    s=outcome_summary(e)
    s.to_csv(args.out/"b60_b90_2r_target_stop_comparison.csv",index=False,encoding="utf-8-sig")
    week=e[e.date.between("2026-11-09","2026-11-13")].copy()
    week.to_csv(args.out/"week_new_b60_b90_breakouts.csv",index=False,encoding="utf-8-sig")
    a=e.groupby(["date","code","base_days"],as_index=False).agg(
        legacy=("gate",lambda q:bool((q=="LEGACY_50E").any())),
        cap=("gate",lambda q:bool((q=="CAP_RELATIVE").any())))
    a["status"]=np.select([a.legacy&a.cap,a.legacy, a.cap],
                          ["BOTH","LEGACY_ONLY","CAP_ONLY"],default="NEITHER")
    a[a.date.between("2026-11-09","2026-11-13")].to_csv(
        args.out/"week_full_gongguri_gate_delta.csv",index=False,encoding="utf-8-sig")
    (args.out/"METHOD.txt").write_text(
        "60-/90-trading-day base, prior 20%+ decline, EMA224 recovery/anchor, "
        "previous 80 sessions at least 38 closes below EMA224, 40-session "
        "per-tier per-gate cooldown. Identical technical V4 base/EMA event "
        "logic and same next-session open structural stop/2R target for both "
        "liquidity arms; a market-cap-size liquidity gate vs old 50e. "
        "Original V4 checks next-day open to confirm stop-distance 2.5-18%, "
        "which is an outcome simulation condition and NOT a pure as-of-close "
        "candidate detection rule. Bars following the signal are not known "
        "when generating a current alert. 20d incomplete trade outcomes are "
        "CENSORED, never assigned timeout. 20/60d gap-aware stop-first fill "
        "with cost 0.4%; historical data can suffer survivorship/provider "
        "bias and stale issuer episodes. Not a paid Dante strategy or "
        "proof of profitability. No orders.\n",encoding="utf-8")
    print("FULL",len(e),"STOCKS",e.code.nunique(),flush=True)
    print("WEEK",len(week),"GATES\n"+week.gate.value_counts().to_string(),flush=True)
    print("BREAKOUT GROUPS\n"+s[s.setup.eq("EMA224_BREAKOUT")].to_string(index=False),flush=True)

if __name__=="__main__":
    main()
