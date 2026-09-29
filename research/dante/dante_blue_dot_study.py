#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grid-search public Blue-dot / 파란점선 approximation families.

Research-only. The proprietary formula is not public.
We test internet hypotheses against dated official/public recaps that explicitly
say the Blue-dot line touched or was near price.

No future data is used: for a forward/right-shifted line, today's plotted value
is computed from a historical Bollinger value.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import FinanceDataReader as fdr

PERIODS=(15,20,25,30,33,35,40,56)
MULTS=(1.5,1.75,2.0,2.25,2.5,3.0)
SHIFTS=(0,5,10,15,20,26,30,35,40)

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

def fallback(code,name,end):
    try:
        q=fdr.DataReader("NAVER:"+code,"2023-01-01",(end+pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
        if q is None or q.empty:return pd.DataFrame()
        q=q.reset_index(); q.columns=[str(c).lower() for c in q.columns]
        q["date"]=pd.to_datetime(q.date,errors="coerce")
        q["code"]=code; q["name"]=name; q["exchange"]="FDR"; q["series_id"]="FDR:"+code
        q["adjusted_close"]=q.close
        return q.dropna(subset=["date","close"]).sort_values("date")
    except Exception:
        return pd.DataFrame()

def prepare(g):
    z=g.sort_values("date").copy().reset_index(drop=True)
    f=(z.adjusted_close/z.close).replace([np.inf,-np.inf],np.nan).ffill().bfill().fillna(1)
    z["ac"]=z.adjusted_close; z["ah"]=z.high*f; z["al"]=z.low*f
    return z

def band(z,p,m,s):
    mid=z.ac.rolling(p,min_periods=p).mean()
    sd=z.ac.rolling(p,min_periods=p).std(ddof=0)
    upper=mid+m*sd
    return upper.shift(s)

def candle_distance_pct(line,low,high,close):
    if pd.isna(line) or line<=0:return np.nan
    if low<=line<=high:return 0.0
    edge=low if line<low else high
    return abs(edge/line-1)*100

def close_distance_pct(line,close):
    if pd.isna(line) or line<=0:return np.nan
    return abs(close/line-1)*100

def case_scores(z):
    rows=[]
    for p in PERIODS:
        for m in MULTS:
            for s in SHIFTS:
                line=band(z,p,m,s)
                if line.isna().all():continue
                i=len(z)-1
                exact_candle=candle_distance_pct(float(line.iloc[i]),float(z.at[i,"al"]),float(z.at[i,"ah"]),float(z.at[i,"ac"]))
                exact_close=close_distance_pct(float(line.iloc[i]),float(z.at[i,"ac"]))
                recent=[]
                recent_close=[]
                for j in range(max(0,i-4),i+1):
                    recent.append(candle_distance_pct(float(line.iloc[j]) if pd.notna(line.iloc[j]) else np.nan,float(z.at[j,"al"]),float(z.at[j,"ah"]),float(z.at[j,"ac"])))
                    recent_close.append(close_distance_pct(float(line.iloc[j]) if pd.notna(line.iloc[j]) else np.nan,float(z.at[j,"ac"])))
                rows.append({
                    "period":p,"mult":m,"shift":s,
                    "exact_candle_dist_pct":exact_candle,
                    "exact_close_dist_pct":exact_close,
                    "min5_candle_dist_pct":np.nanmin(recent) if np.isfinite(recent).any() else np.nan,
                    "min5_close_dist_pct":np.nanmin(recent_close) if np.isfinite(recent_close).any() else np.nan,
                    "line_price":float(line.iloc[i]) if pd.notna(line.iloc[i]) else np.nan,
                    "close":float(z.at[i,"close"]),
                })
    return pd.DataFrame(rows)

def main():
    a=args(); a.out.mkdir(parents=True,exist_ok=True)
    panel=pd.concat([load(a.ko,"KO"),load(a.kq,"KQ")],ignore_index=True)
    cases=pd.read_csv(a.cases,dtype={"code":str}); cases["code"]=cases.code.str.zfill(6); cases["analysis_date"]=pd.to_datetime(cases.analysis_date)
    all_rows=[]; misses=[]
    for c in cases.to_dict("records"):
        q=panel[(panel.code.eq(c["code"]))&(panel.date<=c["analysis_date"])].copy()
        if q.empty:q=fallback(c["code"],c["stock"],pd.Timestamp(c["analysis_date"]))
        if q.empty:
            misses.append({**c,"reason":"not_found"}); continue
        sid=q.groupby("series_id").size().sort_values(ascending=False).index[0]
        z=prepare(q[q.series_id.eq(sid) & (q.date<=c["analysis_date"])])
        if len(z)<100:
            misses.append({**c,"reason":"short_history"}); continue
        scores=case_scores(z)
        for k,v in c.items():scores[k]=v
        scores["market_date"]=z.iloc[-1].date
        all_rows.append(scores)
    if not all_rows:
        return
    detail=pd.concat(all_rows,ignore_index=True)
    detail.to_csv(a.out/"blue_dot_grid_detail.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(misses).to_csv(a.out/"blue_dot_misses.csv",index=False,encoding="utf-8-sig")

    explicit=detail[detail.label.isin(["explicit_touch","near_or_touch"])].copy()
    agg=explicit.groupby(["period","mult","shift"]).agg(
        n=("stock","nunique"),
        median_exact_candle=("exact_candle_dist_pct","median"),
        mean_exact_candle=("exact_candle_dist_pct","mean"),
        median_min5_candle=("min5_candle_dist_pct","median"),
        mean_min5_candle=("min5_candle_dist_pct","mean"),
        within1_exact=("exact_candle_dist_pct",lambda x:float((x<=1).mean())),
        within2_exact=("exact_candle_dist_pct",lambda x:float((x<=2).mean())),
        within3_exact=("exact_candle_dist_pct",lambda x:float((x<=3).mean())),
        within3_min5=("min5_candle_dist_pct",lambda x:float((x<=3).mean())),
        within5_min5=("min5_candle_dist_pct",lambda x:float((x<=5).mean())),
    ).reset_index()
    agg=agg.sort_values(["within3_min5","median_min5_candle","within3_exact","median_exact_candle"],ascending=[False,True,False,True])
    agg.to_csv(a.out/"blue_dot_grid_summary.csv",index=False,encoding="utf-8-sig")

    # Internet hypotheses, evaluated explicitly rather than assumed true.
    wanted={(35,2.0,0):"internet_bb35x2",(20,2.0,26):"internet_bb20x2_shift26"}
    refs=[]
    for key,name in wanted.items():
        q=agg[(agg["period"]==key[0])&(np.isclose(agg["mult"],key[1]))&(agg["shift"]==key[2])].copy()
        if not q.empty:
            r=q.iloc[0].to_dict(); r["hypothesis"]=name; refs.append(r)
    pd.DataFrame(refs).to_csv(a.out/"blue_dot_internet_hypotheses.csv",index=False,encoding="utf-8-sig")

    print(f"cases={len(cases)} matched={detail.stock.nunique()} misses={len(misses)} explicit_or_near={explicit.stock.nunique()}")
    print("\n=== TOP GRID ===")
    print(agg.head(30).to_string(index=False))
    print("\n=== INTERNET HYPOTHESES ===")
    print(pd.DataFrame(refs).to_string(index=False))

if __name__=="__main__":
    main()
