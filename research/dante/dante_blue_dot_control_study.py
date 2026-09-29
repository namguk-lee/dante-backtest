#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Matched-market controls for Blue-dot approximation candidates.

Research only. Compares a few fixed candidate formulas against same-date
KOSPI/KOSDAQ universe touch rates. No proprietary formula is claimed.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

CONFIGS=(
    ("internet_bb35x2",35,2.0,0),
    ("internet_bb20x2_shift26",20,2.0,26),
    ("grid_bb20x2_shift5",20,2.0,5),
    ("grid_bb25x2_25_shift5",25,2.25,5),
    ("noshift_bb35x1_5",35,1.5,0),
)

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--ko",type=Path,required=True)
    p.add_argument("--kq",type=Path,required=True)
    p.add_argument("--cases",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("dante_blue_dot_study"))
    return p.parse_args()

def load(path,ex):
    x=pd.read_parquet(path).copy()
    x["date"]=pd.to_datetime(x.date,errors="coerce")
    x["code"]=x.code.astype(str).str.replace(r"\.0$","",regex=True).str.zfill(6)
    if "exchange" not in x:x["exchange"]=ex
    if "adjusted_close" not in x:x["adjusted_close"]=x.close
    return x.sort_values(["series_id","date"])

def prep(g):
    z=g.sort_values("date").copy().reset_index(drop=True)
    f=(z.adjusted_close/z.close).replace([np.inf,-np.inf],np.nan).ffill().bfill().fillna(1)
    z["ac"]=z.adjusted_close; z["ah"]=z.high*f; z["al"]=z.low*f
    return z

def line(z,p,m,s):
    mid=z.ac.rolling(p,min_periods=p).mean()
    sd=z.ac.rolling(p,min_periods=p).std(ddof=0)
    return (mid+m*sd).shift(s)

def candle_dist(lp,lo,hi):
    if pd.isna(lp) or lp<=0:return np.nan
    if lo<=lp<=hi:return 0.0
    edge=lo if lp<lo else hi
    return abs(edge/lp-1)*100

def metrics_at(z,p,m,s):
    b=line(z,p,m,s)
    i=len(z)-1
    if i<0 or pd.isna(b.iloc[i]):return None
    exact=candle_dist(float(b.iloc[i]),float(z.at[i,"al"]),float(z.at[i,"ah"]))
    recent=[]
    for j in range(max(0,i-4),i+1):
        recent.append(candle_dist(float(b.iloc[j]) if pd.notna(b.iloc[j]) else np.nan,float(z.at[j,"al"]),float(z.at[j,"ah"])))
    finite=[v for v in recent if pd.notna(v)]
    return exact,(min(finite) if finite else np.nan)

def main():
    a=args(); a.out.mkdir(parents=True,exist_ok=True)
    panel=pd.concat([load(a.ko,"KO"),load(a.kq,"KQ")],ignore_index=True)
    cases=pd.read_csv(a.cases,dtype={"code":str})
    cases["code"]=cases.code.astype(str).str.zfill(6)
    cases["analysis_date"]=pd.to_datetime(cases.analysis_date)
    positives=cases[cases.label.isin(["explicit_touch","near_or_touch"])].copy()

    summaries=[]; stocks=[]
    for dt in sorted(positives.analysis_date.unique()):
        dt=pd.Timestamp(dt)
        hist=panel[panel.date<=dt]
        if hist.empty:continue
        market_day=hist.date.max()
        active=set(hist.loc[hist.date.eq(market_day),"series_id"].astype(str))
        groups=hist[hist.series_id.astype(str).isin(active)].groupby("series_id",sort=False)
        for name,p,m,s in CONFIGS:
            vals=[]
            for sid,g in groups:
                z=prep(g)
                r=metrics_at(z,p,m,s)
                if r is None:continue
                last=z.iloc[-1]
                vals.append({
                    "analysis_date":dt,"market_date":market_day,"config":name,
                    "series_id":sid,"code":str(last.code).zfill(6),"name":last.get("name",""),
                    "exact_candle_dist_pct":r[0],"min5_candle_dist_pct":r[1]
                })
            q=pd.DataFrame(vals)
            if q.empty:continue
            summaries.append({
                "analysis_date":dt,"market_date":market_day,"config":name,"n":len(q),
                "exact_touch_pct":float((q.exact_candle_dist_pct<=0).mean()*100),
                "within1_exact_pct":float((q.exact_candle_dist_pct<=1).mean()*100),
                "within3_exact_pct":float((q.exact_candle_dist_pct<=3).mean()*100),
                "within3_min5_pct":float((q.min5_candle_dist_pct<=3).mean()*100),
                "median_exact_dist_pct":float(q.exact_candle_dist_pct.median()),
            })
            stocks.extend(q.to_dict("records"))

    s=pd.DataFrame(summaries)
    allstocks=pd.DataFrame(stocks)
    s.to_csv(a.out/"blue_dot_market_control_summary.csv",index=False,encoding="utf-8-sig")
    allstocks.to_csv(a.out/"blue_dot_market_control_all.csv",index=False,encoding="utf-8-sig")

    labeled=[]
    for c in positives.to_dict("records"):
        for name,p,m,sh in CONFIGS:
            q=allstocks[
                allstocks.analysis_date.eq(pd.Timestamp(c["analysis_date"])) &
                allstocks.config.eq(name) & allstocks.code.eq(c["code"])
            ]
            universe=allstocks[
                allstocks.analysis_date.eq(pd.Timestamp(c["analysis_date"])) &
                allstocks.config.eq(name)
            ]
            if q.empty or universe.empty:
                labeled.append({**c,"config":name,"matched":False}); continue
            r=q.iloc[0]
            exact_pct=float((universe.exact_candle_dist_pct<=r.exact_candle_dist_pct).mean()*100)
            min5_pct=float((universe.min5_candle_dist_pct<=r.min5_candle_dist_pct).mean()*100)
            labeled.append({
                **c,"config":name,"matched":True,
                "exact_candle_dist_pct":r.exact_candle_dist_pct,
                "min5_candle_dist_pct":r.min5_candle_dist_pct,
                "exact_distance_percentile":exact_pct,
                "min5_distance_percentile":min5_pct,
                "universe_n":len(universe),
            })
    lab=pd.DataFrame(labeled)
    lab.to_csv(a.out/"blue_dot_labeled_vs_market.csv",index=False,encoding="utf-8-sig")

    print("=== MARKET CONTROL ===")
    print(s.to_string(index=False))
    print("\n=== LABELED VS MARKET ===")
    print(lab.to_string(index=False))

if __name__=="__main__":
    main()
