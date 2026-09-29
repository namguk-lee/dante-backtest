#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Overlay recent fully-closed KRX sessions onto marcap parquet using pykrx.

The historical panel remains marcap-based. Only dates newer than the marcap
panel are fetched from KRX in daily cross-sections, avoiding per-symbol calls.
"""
from __future__ import annotations
import argparse, math, time
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from pykrx import stock

KST=ZoneInfo("Asia/Seoul")

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--ko",type=Path,required=True)
    p.add_argument("--kq",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("krx_latest"))
    p.add_argument("--target-date",default=None,help="YYYY-MM-DD; defaults to latest fully closed KST session")
    p.add_argument("--retry",type=int,default=3)
    p.add_argument("--sleep",type=float,default=1.0)
    return p.parse_args()

def latest_closed_target(explicit=None):
    if explicit:
        candidate=pd.Timestamp(explicit).date()
    else:
        now=datetime.now(KST)
        candidate=now.date() if now.time()>=dtime(16,10) else now.date()-timedelta(days=1)
    bday=stock.get_nearest_business_day_in_a_week(candidate.strftime("%Y%m%d"),prev=True)
    return pd.Timestamp(str(bday))

def load(path,exchange):
    x=pd.read_parquet(path).copy()
    x["date"]=pd.to_datetime(x["date"],errors="coerce")
    x["code"]=x["code"].astype(str).str.replace(r"\.0$","",regex=True).str.zfill(6)
    if "exchange" not in x:x["exchange"]=exchange
    if "name" not in x:x["name"]=""
    if "adjusted_close" not in x:x["adjusted_close"]=x["close"]
    if "reported_change_pct" not in x:x["reported_change_pct"]=np.nan
    if "adjustment_bridge" not in x:x["adjustment_bridge"]=False
    return x.sort_values(["series_id","date"])

def fetch_day(ds,market,retry,sleep):
    last=None
    for k in range(retry):
        try:
            q=stock.get_market_ohlcv_by_ticker(ds,market=market,alternative=False)
            if q is None:return pd.DataFrame()
            return q.copy()
        except Exception as exc:
            last=exc
            time.sleep(sleep*(k+1))
    raise RuntimeError(f"pykrx fetch failed date={ds} market={market}: {last!r}")

def normalize_krx(q,date,exchange):
    if q.empty:return pd.DataFrame()
    q=q.reset_index()
    code_col="티커" if "티커" in q.columns else q.columns[0]
    ren={
        code_col:"code","시가":"open","고가":"high","저가":"low","종가":"close",
        "거래량":"volume","거래대금":"amount","등락률":"reported_change_pct",
    }
    q=q.rename(columns={k:v for k,v in ren.items() if k in q.columns})
    need=["code","open","high","low","close","volume"]
    if any(c not in q.columns for c in need):
        raise ValueError(f"Unexpected pykrx columns: {list(q.columns)}")
    if "amount" not in q:q["amount"]=q["close"]*q["volume"]
    if "reported_change_pct" not in q:q["reported_change_pct"]=np.nan
    q["code"]=q["code"].astype(str).str.zfill(6)
    for c in ["open","high","low","close","volume","amount","reported_change_pct"]:
        q[c]=pd.to_numeric(q[c],errors="coerce")
    q=q.dropna(subset=need)
    q=q[(q.close>0)&(q.open>0)&(q.high>0)&(q.low>0)]
    q["date"]=pd.Timestamp(date); q["exchange"]=exchange
    return q

def append_exchange(base,market,exchange,target,retry,sleep):
    if base.empty:return base,0
    max_date=pd.Timestamp(base.date.max()).normalize()
    if target<=max_date:return base,0
    latest=(base.sort_values("date").groupby("code",as_index=False).tail(1)
              [["code","series_id","name","close","adjusted_close"]].copy())
    meta=latest.set_index("code").to_dict("index")
    state={c:{"raw":float(v["close"]),"adj":float(v["adjusted_close"])} for c,v in meta.items()
           if pd.notna(v["close"]) and pd.notna(v["adjusted_close"]) and float(v["close"])>0}
    frames=[]; added=0
    for dt in pd.date_range(max_date+pd.Timedelta(days=1),target,freq="D"):
        ds=dt.strftime("%Y%m%d")
        q=normalize_krx(fetch_day(ds,market,retry,sleep),dt,exchange)
        if q.empty:
            print(f"{exchange} {dt.date()} empty/holiday",flush=True); continue
        rows=[]
        for rec in q.to_dict("records"):
            code=rec["code"]
            if code not in meta or code not in state:
                continue  # no EMA448 warmup anyway
            prev=state[code]
            raw_ret=rec["close"]/prev["raw"]-1 if prev["raw"]>0 else np.nan
            rr=rec.get("reported_change_pct",np.nan)
            reported=rr/100.0 if pd.notna(rr) else np.nan
            bridge=bool(pd.notna(reported) and abs(reported)<=0.40 and pd.notna(raw_ret) and abs(raw_ret-reported)>0.03)
            eff=reported if bridge else raw_ret
            if not pd.notna(eff): eff=0.0
            eff=float(np.clip(eff,-0.95,5.0))
            adj=prev["adj"]*(1.0+eff)
            m=meta[code]
            rows.append({
                "series_id":m["series_id"],"code":code,"exchange":exchange,"name":m.get("name",""),
                "date":dt,"open":rec["open"],"high":rec["high"],"low":rec["low"],"close":rec["close"],
                "adjusted_close":adj,"volume":rec["volume"],"amount":rec["amount"],
                "reported_change_pct":rec.get("reported_change_pct",np.nan),"adjustment_bridge":bridge,
            })
            state[code]={"raw":float(rec["close"]),"adj":float(adj)}
        if rows:
            z=pd.DataFrame(rows); frames.append(z); added+=len(z)
            print(f"{exchange} {dt.date()} added={len(z):,}",flush=True)
    if not frames:return base,0
    out=pd.concat([base,*frames],ignore_index=True,sort=False)
    out=out.sort_values(["series_id","date"]).drop_duplicates(["series_id","date"],keep="last")
    return out,added

def main():
    a=parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    ko=load(a.ko,"KO"); kq=load(a.kq,"KQ")
    target=latest_closed_target(a.target_date)
    print(f"target_closed_session={target.date()} base_KO={ko.date.max().date()} base_KQ={kq.date.max().date()}",flush=True)
    ko2,nko=append_exchange(ko,"KOSPI","KO",target,a.retry,a.sleep)
    kq2,nkq=append_exchange(kq,"KOSDAQ","KQ",target,a.retry,a.sleep)
    ko2.to_parquet(a.out/"ko_eod.parquet",index=False)
    kq2.to_parquet(a.out/"kq_eod.parquet",index=False)
    ko_last=pd.Timestamp(ko2.date.max()).normalize(); kq_last=pd.Timestamp(kq2.date.max()).normalize()
    print(f"overlay added KO={nko:,} KQ={nkq:,}; final KO={ko_last.date()} KQ={kq_last.date()}",flush=True)
    if ko_last<target or kq_last<target:
        raise SystemExit(f"Latest overlay incomplete: target={target.date()} KO={ko_last.date()} KQ={kq_last.date()}")
if __name__=="__main__":
    main()
