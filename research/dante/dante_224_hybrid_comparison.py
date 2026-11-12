#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare three transparent Dante-style research paths on the same KRX panel.

1) OLD_224_SETTLE
   User-provided long-decline -> EMA224 settlement idea, with two correctness fixes:
   - settlement date is the FALSE->TRUE transition, not any later TRUE day
   - decline is chronological max drawdown, not unordered max(high)/min(low)

2) EVENT_FIRST_E3
   Reuses the repository's current E3_CLASSIC_PULLBACK implementation exactly.

3) HYBRID_EVENT_224
   OLD_224_SETTLE plus a qualifying strong price/volume energy event in the prior
   60 sessions. This tests the user's proposed "energy first, Dante structure next"
   workflow without tuning extra parameters to the validation set.

Research only. No proprietary Dante indicator is reproduced.
"""
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd

from dante_event_first_backtest import scan_one as event_scan_one

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--ko",type=Path,required=True)
    p.add_argument("--kq",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("hybrid_compare"))
    p.add_argument("--start",default="2018-01-01")
    p.add_argument("--cost-bps",type=float,default=50.0)
    return p.parse_args()

def load(path,ex):
    x=pd.read_parquet(path).copy()
    x["date"]=pd.to_datetime(x["date"],errors="coerce")
    if "exchange" not in x:x["exchange"]=ex
    if "adjusted_close" not in x:x["adjusted_close"]=x["close"]
    if "amount" not in x:x["amount"]=x["close"]*x["volume"]
    return x.sort_values(["series_id","date"])

def split(d):
    y=pd.Timestamp(d).year
    return "TRAIN" if y<=2021 else "VALID" if y<=2024 else "TEST"

def feat(g):
    z=g.sort_values("date").copy().reset_index(drop=True)
    factor=(z.adjusted_close/z.close).replace([np.inf,-np.inf],np.nan).ffill().bfill().fillna(1.0)
    z["ao"]=z.open*factor; z["ah"]=z.high*factor; z["al"]=z.low*factor; z["ac"]=z.adjusted_close
    for n in (56,112,224,448):
        z[f"ema{n}"]=z.ac.ewm(span=n,adjust=False,min_periods=n).mean()
    z["amount"]=pd.to_numeric(z.amount,errors="coerce")
    z["v20_prev"]=z.volume.shift(1).rolling(20,min_periods=20).mean()
    z["amt20_prev"]=z.amount.shift(1).rolling(20,min_periods=20).mean()
    z["vr"]=np.where(z.v20_prev>0,z.volume/z.v20_prev,np.nan)
    z["amount_ratio"]=np.where(z.amt20_prev>0,z.amount/z.amt20_prev,np.nan)
    z["ret1"]=z.ac.pct_change(fill_method=None)
    z["body"]=(z.ac-z.ao)/z.ao

    z["above224"]=z.ac>=z.ema224
    z["below224"]=z.ac<z.ema224
    z["above224_count5"]=z.above224.rolling(5,min_periods=5).sum()
    z["settle224"]=(
        (z.above224_count5>=3) &
        (z.ac>=z.ema224*.99) &
        (z.ac<=z.ema224*1.08)
    )
    z["settle224_start"]=z.settle224 & (~z.settle224.shift(1).fillna(False))

    z["below224_count_prev"]=z.below224.shift(5).rolling(120,min_periods=120).sum()
    z["reverse224"]=(z.ema56<z.ema112)&(z.ema112<z.ema224)
    z["reverse224_count_prev"]=z.reverse224.shift(5).rolling(90,min_periods=90).sum()

    z["energy_event"]=(
        (z.ret1>=.05) &
        (z.body>=.035) &
        (z.vr>=2.0) &
        (z.amt20_prev>=5_000_000_000)
    ).fillna(False)
    return z

def chronological_decline_pct(z,signal_i,lookback=260):
    lo=max(0,signal_i-lookback+1)
    w=z.iloc[lo:signal_i+1]
    if len(w)<40:return np.nan
    running_high=w.ah.cummax()
    dd=w.al/running_high-1
    if dd.dropna().empty:return np.nan
    return float(-dd.min()*100.0)

def recent_support_ok(z,i):
    lo=max(0,i-2)
    w=z.iloc[lo:i+1]
    if len(w)<3:return False
    return bool((w.al>=w.ema224*.97).all())

def recent_energy(z,i,lookback=60):
    lo=max(0,i-lookback+1)
    ids=np.flatnonzero(z.energy_event.iloc[lo:i+1].to_numpy())
    if not len(ids):return None
    return lo+int(ids[-1])

def add_forward(rows,z,i,strategy,cost,event_i=None):
    entry_i=i+1
    if entry_i>=len(z):return
    entry=float(z.at[entry_i,"ao"])
    if not math.isfinite(entry) or entry<=0:return
    rec={
        "strategy":strategy,
        "series_id":z.at[i,"series_id"],
        "exchange":z.at[i,"exchange"],
        "signal_date":z.at[i,"date"],
        "entry_date":z.at[entry_i,"date"],
        "entry":entry,
        "split":split(z.at[i,"date"]),
        "decline_pct":chronological_decline_pct(z,i),
        "below224_days":float(z.at[i,"below224_count_prev"]) if pd.notna(z.at[i,"below224_count_prev"]) else np.nan,
        "reverse224_days":float(z.at[i,"reverse224_count_prev"]) if pd.notna(z.at[i,"reverse224_count_prev"]) else np.nan,
        "dist224":float(z.at[i,"ac"]/z.at[i,"ema224"]-1) if pd.notna(z.at[i,"ema224"]) else np.nan,
        "ema448_headroom":float(z.at[i,"ema448"]/z.at[i,"ac"]-1) if pd.notna(z.at[i,"ema448"]) and z.at[i,"ac"]>0 else np.nan,
        "current_amount":float(z.at[i,"amount"]) if pd.notna(z.at[i,"amount"]) else np.nan,
        "event_date":z.at[event_i,"date"] if event_i is not None else pd.NaT,
        "event_return":float(z.at[event_i,"ret1"]) if event_i is not None and pd.notna(z.at[event_i,"ret1"]) else np.nan,
        "event_volume_ratio":float(z.at[event_i,"vr"]) if event_i is not None and pd.notna(z.at[event_i,"vr"]) else np.nan,
    }
    for h in (20,60):
        j=entry_i+h
        if j<len(z):
            rec[f"ret{h}"]=float(z.at[j,"ac"]/entry-1-cost)
        else:
            rec[f"ret{h}"]=np.nan
    rows.append(rec)

def scan_settle(g,start,cost):
    z=feat(g)
    rows=[]
    if len(z)<520:return rows,z
    start_i=max(500,int(z.index[z.date>=start][0]) if (z.date>=start).any() else len(z))
    last_signal={"OLD_224_SETTLE":-999,"OLD_224_SETTLE_LIQUID":-999,"HYBRID_EVENT_224":-999}
    for i in range(start_i,len(z)-62):
        if not bool(z.at[i,"settle224_start"]):continue
        if pd.isna(z.at[i,"below224_count_prev"]) or float(z.at[i,"below224_count_prev"])<80:continue
        if pd.isna(z.at[i,"reverse224_count_prev"]) or float(z.at[i,"reverse224_count_prev"])<40:continue
        if not recent_support_ok(z,i):continue
        if pd.isna(z.at[i,"amount"]) or float(z.at[i,"amount"])<100_000_000:continue
        decline=chronological_decline_pct(z,i)
        if not math.isfinite(decline) or decline<30:continue

        if i-last_signal["OLD_224_SETTLE"]>=40:
            add_forward(rows,z,i,"OLD_224_SETTLE",cost)
            last_signal["OLD_224_SETTLE"]=i

        if float(z.at[i,"amount"])>=5_000_000_000 and i-last_signal["OLD_224_SETTLE_LIQUID"]>=40:
            add_forward(rows,z,i,"OLD_224_SETTLE_LIQUID",cost)
            last_signal["OLD_224_SETTLE_LIQUID"]=i

        e=recent_energy(z,i,60)
        if e is not None and i-last_signal["HYBRID_EVENT_224"]>=40:
            add_forward(rows,z,i,"HYBRID_EVENT_224",cost,event_i=e)
            last_signal["HYBRID_EVENT_224"]=i
    return rows,z

def summarize(ev):
    out=[]
    for (sp,strategy),q in ev.groupby(["split","strategy"],dropna=False):
        rec={"split":sp,"strategy":strategy,"n":len(q)}
        for h in (20,60):
            s=q[f"ret{h}"].dropna()
            rec[f"n{h}"]=len(s)
            rec[f"mean{h}"]=s.mean() if len(s) else np.nan
            rec[f"median{h}"]=s.median() if len(s) else np.nan
            rec[f"win{h}"]=(s>0).mean() if len(s) else np.nan
            rec[f"p25_{h}"]=s.quantile(.25) if len(s) else np.nan
            rec[f"p75_{h}"]=s.quantile(.75) if len(s) else np.nan
        out.append(rec)
    return pd.DataFrame(out).sort_values(["split","strategy"])

def yearly(ev):
    x=ev.copy()
    x["year"]=pd.to_datetime(x.signal_date,errors="coerce").dt.year
    out=[]
    for (yr,strategy),q in x.groupby(["year","strategy"],dropna=False):
        rec={"year":int(yr),"strategy":strategy,"n":len(q)}
        for h in (20,60):
            s=q[f"ret{h}"].dropna()
            rec[f"mean{h}"]=s.mean() if len(s) else np.nan
            rec[f"median{h}"]=s.median() if len(s) else np.nan
            rec[f"win{h}"]=(s>0).mean() if len(s) else np.nan
        out.append(rec)
    return pd.DataFrame(out).sort_values(["year","strategy"])

def main():
    a=parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    panel=pd.concat([load(a.ko,"KO"),load(a.kq,"KQ")],ignore_index=True)
    start=pd.Timestamp(a.start); cost=a.cost_bps/10000.0
    rows=[]; total=panel.series_id.nunique()
    for n,(sid,g) in enumerate(panel.groupby("series_id",sort=False),1):
        settle_rows,_=scan_settle(g,start,cost)
        rows.extend(settle_rows)

        # Exact repository Event-First E3, on the same symbol episode and cost.
        if len(g)>=560:
            erows=event_scan_one(g,start,cost)
            for r in erows:
                if r.get("kind")!="E3_CLASSIC_PULLBACK":continue
                rows.append({
                    "strategy":"EVENT_FIRST_E3",
                    "series_id":r.get("series_id"),"exchange":r.get("exchange"),
                    "signal_date":r.get("signal_date"),"entry_date":r.get("entry_date"),
                    "entry":r.get("entry"),"split":r.get("split"),
                    "ret20":r.get("ret20"),"ret60":r.get("ret60"),
                    "decline_pct":np.nan,"below224_days":np.nan,"reverse224_days":np.nan,
                    "dist224":np.nan,"ema448_headroom":r.get("signal_headroom448",np.nan),
                    "current_amount":np.nan,"event_date":r.get("event_date"),
                    "event_return":r.get("event_return",np.nan),
                    "event_volume_ratio":r.get("event_volume_ratio",np.nan),
                })
        if n%250==0 or n==total:
            print(f"series {n}/{total} rows={len(rows):,}",flush=True)

    ev=pd.DataFrame(rows)
    ev.to_csv(a.out/"strategy_events.csv",index=False,encoding="utf-8-sig")
    sm=summarize(ev) if not ev.empty else pd.DataFrame()
    yr=yearly(ev) if not ev.empty else pd.DataFrame()
    sm.to_csv(a.out/"strategy_summary.csv",index=False,encoding="utf-8-sig")
    yr.to_csv(a.out/"strategy_yearly.csv",index=False,encoding="utf-8-sig")

    print("\n=== OLD vs EVENT-FIRST vs HYBRID ===")
    if not sm.empty:
        show=sm.copy()
        for c in [x for x in show.columns if x.startswith(("mean","median","win","p25_","p75_"))]:
            show[c]=(show[c]*100).round(2)
        print(show.to_string(index=False))

    print("\n=== YEARLY ===")
    if not yr.empty:
        y=yr.copy()
        for c in [x for x in y.columns if x.startswith(("mean","median","win"))]:
            y[c]=(y[c]*100).round(2)
        print(y.to_string(index=False))

if __name__=="__main__":
    main()
