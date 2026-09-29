#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Refresh only plausible Dante candidates with recent Naver/FDR daily bars.

Why:
- marcap is excellent for survivorship-aware history but may lag a few sessions.
- KRX bulk endpoints may require login.
- EMA224/448 change slowly, so we prefilter broadly from the last complete
  marcap session, then refresh only plausible candidates via FinanceDataReader.
"""
from __future__ import annotations
import argparse, math, random, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import FinanceDataReader as fdr

KST=ZoneInfo("Asia/Seoul")

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--ko",type=Path,required=True); p.add_argument("--kq",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("krx_latest"))
    p.add_argument("--workers",type=int,default=6); p.add_argument("--retries",type=int,default=3)
    p.add_argument("--min-turnover",type=float,default=1_000_000_000)
    p.add_argument("--below-min",type=int,default=50)
    p.add_argument("--dist-low",type=float,default=-0.20); p.add_argument("--dist-high",type=float,default=0.30)
    p.add_argument("--max-candidates",type=int,default=500)
    p.add_argument("--target-date",default=None)
    return p.parse_args()

def load(path,ex):
    x=pd.read_parquet(path).copy(); x["date"]=pd.to_datetime(x.date,errors="coerce")
    x["code"]=x.code.astype(str).str.replace(r"\.0$","",regex=True).str.zfill(6)
    if "exchange" not in x:x["exchange"]=ex
    if "name" not in x:x["name"]=""
    if "adjusted_close" not in x:x["adjusted_close"]=x["close"]
    if "amount" not in x:x["amount"]=x["close"]*x["volume"]
    if "reported_change_pct" not in x:x["reported_change_pct"]=np.nan
    if "adjustment_bridge" not in x:x["adjustment_bridge"]=False
    return x.sort_values(["series_id","date"])

def indicators(g):
    g=g.sort_values("date").copy()
    ac=pd.to_numeric(g.adjusted_close,errors="coerce")
    g["ema112"]=ac.ewm(span=112,adjust=False,min_periods=112).mean()
    g["ema224"]=ac.ewm(span=224,adjust=False,min_periods=224).mean()
    g["ema448"]=ac.ewm(span=448,adjust=False,min_periods=448).mean()
    g["turn20"]=pd.to_numeric(g.amount,errors="coerce").rolling(20,min_periods=20).mean()
    g["below80"]=(ac<g.ema224).shift(1).rolling(80,min_periods=80).sum()
    return g

def prefilter(panel,a):
    base_date=panel.date.max(); rows=[]
    active=panel[panel.date.eq(base_date)]
    active_ids=set(active.series_id.astype(str))
    for sid,g in panel[panel.series_id.astype(str).isin(active_ids)].groupby("series_id",sort=False):
        z=indicators(g); r=z.iloc[-1]
        vals=[r.ema112,r.ema224,r.ema448,r.turn20,r.adjusted_close]
        if not all(pd.notna(v) and math.isfinite(float(v)) for v in vals):continue
        rev=bool(r.ema112<r.ema224<r.ema448)
        dist=float(r.adjusted_close/r.ema224-1)
        if not rev or r.below80<a.below_min or r.turn20<a.min_turnover or not(a.dist_low<=dist<=a.dist_high):continue
        rows.append({"series_id":sid,"code":str(r.code).zfill(6),"exchange":r.exchange,"name":r.get("name",""),
                     "base_date":base_date,"dist224":dist,"below80":int(r.below80),"turn20":float(r.turn20),
                     "ema224":float(r.ema224)})
    q=pd.DataFrame(rows)
    if q.empty:return q
    q["priority"]=q.dist224.abs()-np.minimum(np.log10(np.maximum(q.turn20,1)/a.min_turnover),2)*0.01
    return q.sort_values(["priority","turn20"],ascending=[True,False]).head(a.max_candidates).reset_index(drop=True)

def fetch_one(rec,start,end,retries):
    last=None
    for k in range(retries):
        try:
            q=fdr.DataReader(rec["code"],start.strftime("%Y-%m-%d"),(end+pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
            if q is None or q.empty:return rec["series_id"],pd.DataFrame(),"empty"
            q=q.reset_index(); lower={str(c).lower():c for c in q.columns}
            ren={}
            for canon,aliases in {"date":["date","index"],"open":["open"],"high":["high"],"low":["low"],"close":["close"],"volume":["volume"],"change":["change"],"amount":["amount"]}.items():
                for al in aliases:
                    if al in lower:ren[lower[al]]=canon;break
            q=q.rename(columns=ren)
            need=["date","open","high","low","close","volume"]
            if any(c not in q for c in need):raise ValueError(f"columns={list(q.columns)}")
            q["date"]=pd.to_datetime(q.date,errors="coerce")
            for c in ["open","high","low","close","volume"]+(["change"] if "change" in q else [])+(["amount"] if "amount" in q else []):
                q[c]=pd.to_numeric(q[c],errors="coerce")
            q=q.dropna(subset=need); q=q[(q.date>start)&(q.date<=end)]
            q=q[(q.open>0)&(q.high>0)&(q.low>0)&(q.close>0)]
            if "amount" not in q:q["amount"]=q.close*q.volume
            return rec["series_id"],q.sort_values("date"),None
        except Exception as exc:
            last=repr(exc); time.sleep((k+1)*0.8+random.random()*0.3)
    return rec["series_id"],pd.DataFrame(),last

def stitch(base,cand,results):
    bysid={r.series_id:r for _,r in cand.iterrows()}; frames=[]; logs=[]
    for sid,q,err in results:
        meta=bysid[sid]; hist=base[base.series_id.eq(sid)].sort_values("date")
        if hist.empty or q.empty:
            logs.append({"series_id":sid,"code":meta.code,"rows":0,"last_date":pd.NaT,"error":err or "empty"}); continue
        prev=hist.iloc[-1]; prev_raw=float(prev.close); prev_adj=float(prev.adjusted_close); rows=[]
        for rec in q.to_dict("records"):
            raw_ret=float(rec["close"]/prev_raw-1) if prev_raw>0 else np.nan
            ch=rec.get("change",np.nan)
            reported=float(ch) if pd.notna(ch) and abs(float(ch))<1 else np.nan
            # FDR/Naver Change is normally fractional return, e.g. 0.0123.
            bridge=bool(pd.notna(reported) and pd.notna(raw_ret) and abs(raw_ret-reported)>0.03)
            eff=reported if bridge else raw_ret
            if not pd.notna(eff):eff=0.0
            eff=float(np.clip(eff,-0.95,5.0)); adj=prev_adj*(1+eff)
            rows.append({"series_id":sid,"code":meta.code,"exchange":meta.exchange,"name":meta["name"],
                         "date":rec["date"],"open":rec["open"],"high":rec["high"],"low":rec["low"],"close":rec["close"],
                         "adjusted_close":adj,"volume":rec["volume"],"amount":rec["amount"],
                         "reported_change_pct":reported*100 if pd.notna(reported) else np.nan,"adjustment_bridge":bridge})
            prev_raw=float(rec["close"]); prev_adj=adj
        z=pd.DataFrame(rows); frames.append(z)
        logs.append({"series_id":sid,"code":meta.code,"rows":len(z),"last_date":z.date.max() if not z.empty else pd.NaT,"error":err})
    out=pd.concat([base,*frames],ignore_index=True,sort=False) if frames else base.copy()
    out=out.sort_values(["series_id","date"]).drop_duplicates(["series_id","date"],keep="last")
    return out,pd.DataFrame(logs)

def main():
    a=parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    ko=load(a.ko,"KO"); kq=load(a.kq,"KQ"); panel=pd.concat([ko,kq],ignore_index=True)
    base_date=pd.Timestamp(panel.date.max()).normalize()
    if a.target_date:target=pd.Timestamp(a.target_date).normalize()
    else:target=pd.Timestamp(datetime.now(KST).date()-timedelta(days=1))
    cand=prefilter(panel,a); cand.to_csv(a.out/"refresh_candidates.csv",index=False,encoding="utf-8-sig")
    print(f"base={base_date.date()} target<={target.date()} refresh_candidates={len(cand):,}",flush=True)
    results=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futs=[pool.submit(fetch_one,r,base_date,target,a.retries) for r in cand.to_dict("records")]
        for i,f in enumerate(as_completed(futs),1):
            results.append(f.result())
            if i%50==0 or i==len(futs):print(f"refresh {i}/{len(futs)}",flush=True)
    refreshed,log=stitch(panel,cand,results); log.to_csv(a.out/"refresh_log.csv",index=False,encoding="utf-8-sig")
    ko2=refreshed[refreshed.exchange.eq("KO")]; kq2=refreshed[refreshed.exchange.eq("KQ")]
    ko2.to_parquet(a.out/"ko_eod.parquet",index=False); kq2.to_parquet(a.out/"kq_eod.parquet",index=False)
    ok=log[log.rows>0] if not log.empty else log
    latest=pd.to_datetime(ok.last_date,errors="coerce").max() if not ok.empty else pd.NaT
    errors=int(log.error.notna().sum()) if not log.empty else 0
    print(f"refreshed_ok={len(ok):,}/{len(cand):,} errors={errors:,} latest={latest.date() if pd.notna(latest) else None}",flush=True)
    if len(cand)==0 or len(ok)<max(1,int(len(cand)*0.70)):
        raise SystemExit("Too few recent candidate histories refreshed")
if __name__=="__main__":main()
