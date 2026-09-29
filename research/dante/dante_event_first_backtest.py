#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Historical validation for the event-first Dante hypothesis.

Question:
Does filtering the universe by a prior strong price/volume event, then waiting
for EMA224 recovery and a lower-volume pullback, improve forward returns?

Transparent approximation only; no proprietary indicators are reproduced.
"""
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--ko",type=Path,required=True); p.add_argument("--kq",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("event_backtest"))
    p.add_argument("--start",default="2018-01-01")
    p.add_argument("--cost-bps",type=float,default=50.0)
    return p.parse_args()

def load(path,ex):
    x=pd.read_parquet(path).copy()
    x["date"]=pd.to_datetime(x.date,errors="coerce")
    if "exchange" not in x:x["exchange"]=ex
    if "adjusted_close" not in x:x["adjusted_close"]=x["close"]
    if "amount" not in x:x["amount"]=x["close"]*x["volume"]
    return x.sort_values(["series_id","date"])

def feat(g):
    g=g.sort_values("date").copy()
    f=(g.adjusted_close/g.close).replace([np.inf,-np.inf],np.nan).ffill().bfill().fillna(1.0)
    g["ao"]=g.open*f; g["ah"]=g.high*f; g["al"]=g.low*f; g["ac"]=g.adjusted_close
    for n in (112,224,448):g[f"ema{n}"]=g.ac.ewm(span=n,adjust=False,min_periods=n).mean()
    g["v20"]=g.volume.rolling(20,min_periods=20).mean()
    g["amt20"]=g.amount.rolling(20,min_periods=20).mean()
    g["vr"]=g.volume/g.v20
    g["ret1"]=g.ac.pct_change(fill_method=None)
    g["body"]=(g.ac-g.ao)/g.ao
    rng=(g.ah-g.al).replace(0,np.nan)
    g["close_pos"]=(g.ac-g.al)/rng
    g["upper_wick"]=(g.ah-np.maximum(g.ao,g.ac))/rng
    g["cross224"]=(g.ac>g.ema224)&(g.ac.shift(1)<=g.ema224.shift(1))
    g["below80"]=(g.ac<g.ema224).shift(1).rolling(80,min_periods=80).sum()
    return g.reset_index(drop=True)

def split(d):
    y=pd.Timestamp(d).year
    return "TRAIN" if y<=2021 else "VALID" if y<=2024 else "TEST"

def add_event(rows,z,kind,signal_i,event_i,cost):
    entry_i=signal_i+1
    if entry_i>=len(z):return
    entry=float(z.at[entry_i,"ao"])
    if not math.isfinite(entry) or entry<=0:return
    rec={"series_id":z.at[signal_i,"series_id"],"exchange":z.at[signal_i,"exchange"],
         "kind":kind,"event_date":z.at[event_i,"date"],"signal_date":z.at[signal_i,"date"],
         "entry_date":z.at[entry_i,"date"],"entry":entry,"split":split(z.at[signal_i,"date"]),
         "event_quality":"CLEAN_BREAKOUT" if (z.at[event_i,"close_pos"]>=.65 and z.at[event_i,"upper_wick"]<=.35) else "WICK_ENERGY"}
    for h in (20,60):
        j=entry_i+h
        if j<len(z):
            gross=float(z.at[j,"ac"]/entry-1)
            rec[f"ret{h}"]=gross-cost
        else:rec[f"ret{h}"]=np.nan
    rows.append(rec)

def scan_one(g,start,cost):
    z=feat(g)
    rows=[]; cooldown={"E0_EVENT":-999,"E1_BOWL_EVENT":-999,"E2_224_RECOVERY":-999,"E3_CLASSIC_PULLBACK":-999}
    start_i=max(500,int(z.index[z.date>=start][0]) if (z.date>=start).any() else len(z))
    strong_mask=(
        (z.ret1>=.05) & (z.body>=.035) & (z.vr>=2) & (z.amt20>=5_000_000_000)
    ).fillna(False).to_numpy()
    event_idx=np.flatnonzero(strong_mask)
    event_idx=event_idx[(event_idx>=start_i)&(event_idx<len(z)-62)]
    for i in event_idx:
        r=z.iloc[i]
        if i-cooldown["E0_EVENT"]>=40:
            add_event(rows,z,"E0_EVENT",i,i,cost); cooldown["E0_EVENT"]=i

        long_below=bool(r.below80>=60)
        reverse=bool(r.ema112<r.ema224<r.ema448) if all(pd.notna(r[x]) for x in ["ema112","ema224","ema448"]) else False
        if not(long_below and reverse):continue
        if i-cooldown["E1_BOWL_EVENT"]>=40:
            add_event(rows,z,"E1_BOWL_EVENT",i,i,cost); cooldown["E1_BOWL_EVENT"]=i

        anchor=float(r.ao)
        # Stage 2: first EMA224 recovery after the event, while anchor remains alive.
        cross_i=None
        for j in range(i,min(i+21,len(z)-1)):
            post=z.iloc[i:j+1]
            if int((post.ac<anchor).sum())>1:break
            if bool(z.at[j,"cross224"]):
                cross_i=j;break
        if cross_i is None:continue
        headroom=float(z.at[cross_i,"ema448"]/z.at[cross_i,"ac"]-1) if z.at[cross_i,"ac"]>0 else np.nan
        gap=float(abs(z.at[cross_i,"ema224"]-z.at[cross_i,"ema112"])/z.at[cross_i,"ema224"])
        if not(0.05<=headroom<=0.35 and gap<=.08):continue
        if cross_i-cooldown["E2_224_RECOVERY"]>=40:
            add_event(rows,z,"E2_224_RECOVERY",cross_i,i,cost); cooldown["E2_224_RECOVERY"]=cross_i

        # Stage 3: lower-volume pullback that still holds EMA224 and anchor.
        pull_i=None
        event_vol=float(r.volume)
        post_high=-np.inf
        for j in range(cross_i+1,min(cross_i+16,len(z)-1)):
            post_high=max(post_high,float(z.at[j,"ah"]))
            if float(z.at[j,"ac"])<anchor:break
            dist=float(z.at[j,"ac"]/z.at[j,"ema224"]-1)
            dd=float(z.at[j,"ac"]/post_high-1) if post_high>0 else 0
            volcool=float(z.at[j,"volume"])<=event_vol*.60
            if 0<=dist<=.08 and -.20<=dd<=-.03 and volcool:
                pull_i=j;break
        if pull_i is not None and pull_i-cooldown["E3_CLASSIC_PULLBACK"]>=40:
            add_event(rows,z,"E3_CLASSIC_PULLBACK",pull_i,i,cost); cooldown["E3_CLASSIC_PULLBACK"]=pull_i
    return rows

def summarize(ev):
    out=[]
    regimes=[
        ("ALL",ev),
        ("BREADTH20_50",ev[ev.breadth20>=.50]),
        ("BREADTH20_60",ev[ev.breadth20>=.60]),
        ("BOTH50",ev[(ev.breadth20>=.50)&(ev.breadth60>=.50)]),
        ("BOTH60",ev[(ev.breadth20>=.60)&(ev.breadth60>=.60)]),
    ]
    for regime,base in regimes:
        for (sp,k),q in base.groupby(["split","kind"],dropna=False):
            rec={"regime":regime,"split":sp,"kind":k,"n":len(q)}
            for h in (20,60):
                s=q[f"ret{h}"].dropna()
                rec[f"n{h}"]=len(s); rec[f"mean{h}"]=s.mean() if len(s) else np.nan
                rec[f"median{h}"]=s.median() if len(s) else np.nan
                rec[f"win{h}"]=(s>0).mean() if len(s) else np.nan
            out.append(rec)
    return pd.DataFrame(out).sort_values(["regime","split","kind"])

def summarize_yearly(ev):
    out=[]
    if ev.empty:return pd.DataFrame()
    x=ev.copy()
    x["year"]=pd.to_datetime(x["signal_date"],errors="coerce").dt.year
    groups=[
        ("YEAR_ALL",["year","kind"]),
        ("YEAR_EXCHANGE",["year","exchange","kind"]),
        ("YEAR_QUALITY",["year","event_quality","kind"]),
    ]
    for view,keys in groups:
        for vals,q in x.groupby(keys,dropna=False):
            if not isinstance(vals,tuple):vals=(vals,)
            rec={"view":view}
            for k,v in zip(keys,vals):rec[k]=v
            rec["n"]=len(q)
            for h in (20,60):
                s=q[f"ret{h}"].dropna()
                rec[f"n{h}"]=len(s)
                rec[f"mean{h}"]=s.mean() if len(s) else np.nan
                rec[f"median{h}"]=s.median() if len(s) else np.nan
                rec[f"win{h}"]=(s>0).mean() if len(s) else np.nan
            out.append(rec)
    return pd.DataFrame(out)

def main():
    a=parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    panel=pd.concat([load(a.ko,"KO"),load(a.kq,"KQ")],ignore_index=True)
    panel=panel.sort_values(["series_id","date"]).reset_index(drop=True)
    gret=panel.groupby("series_id",sort=False)["adjusted_close"]
    panel["mret20"]=gret.pct_change(20,fill_method=None)
    panel["mret60"]=gret.pct_change(60,fill_method=None)
    b=panel.dropna(subset=["mret20","mret60"]).copy()
    b["up20"]=(b["mret20"]>0).astype(float)
    b["up60"]=(b["mret60"]>0).astype(float)
    breadth=(b.groupby(["exchange","date"],as_index=False)[["up20","up60"]].mean()
               .rename(columns={"date":"signal_date","up20":"breadth20","up60":"breadth60"}))
    start=pd.Timestamp(a.start); cost=a.cost_bps/10000.0
    rows=[]; total=panel.series_id.nunique()
    for n,(sid,g) in enumerate(panel.groupby("series_id",sort=False),1):
        if len(g)>=560:rows.extend(scan_one(g,start,cost))
        if n%250==0 or n==total:print(f"series {n}/{total} events={len(rows):,}",flush=True)
    ev=pd.DataFrame(rows)
    if not ev.empty:
        ev["signal_date"]=pd.to_datetime(ev["signal_date"],errors="coerce")
        ev=ev.merge(breadth,on=["exchange","signal_date"],how="left")
    ev.to_csv(a.out/"event_first_historical_events.csv",index=False,encoding="utf-8-sig")
    sm=summarize(ev) if not ev.empty else pd.DataFrame()
    sm.to_csv(a.out/"event_first_summary.csv",index=False,encoding="utf-8-sig")
    yr=summarize_yearly(ev)
    yr.to_csv(a.out/"event_first_yearly.csv",index=False,encoding="utf-8-sig")
    print("\n=== EVENT_FIRST SUMMARY (returns net of one round-trip cost assumption) ===")
    if not sm.empty:
        show=sm.copy()
        for c in ["mean20","median20","win20","mean60","median60","win60"]:
            if c in show:show[c]=(show[c]*100).round(2)
        print(show.to_string(index=False))
if __name__=="__main__":main()