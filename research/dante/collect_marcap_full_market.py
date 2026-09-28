#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build point-in-time KOSPI/KOSDAQ daily data from FinanceData/marcap.

marcap is a daily historical cross-section, so stocks that later delisted remain
in historical files. Raw exchange prices are converted to a research-adjusted
close: ordinary days use raw close returns; when raw return materially disagrees
with KRX-reported ChagesRatio, the reported return bridges the discontinuity.
This is a research adjustment, not an official vendor adjustment factor.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import urllib.request
import numpy as np
import pandas as pd

RAW_BASE = "https://raw.githubusercontent.com/FinanceData/marcap/master/data"

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--start",default="2016-01-01")
    p.add_argument("--end",default=None)
    p.add_argument("--out",type=Path,default=Path("krx_data"))
    p.add_argument("--cache",type=Path,default=Path("marcap_cache"))
    p.add_argument("--refresh-end-year",action="store_true")
    p.add_argument("--episode-gap-days",type=int,default=365)
    p.add_argument("--name-change-gap-days",type=int,default=30)
    p.add_argument("--corp-action-diff",type=float,default=0.03)
    return p.parse_args()

def download_year(year,cache,refresh):
    cache.mkdir(parents=True,exist_ok=True)
    dest=cache/f"marcap-{year}.parquet"
    if dest.exists() and dest.stat().st_size>100_000 and not refresh:
        return dest
    tmp=dest.with_suffix(".parquet.tmp")
    url=f"{RAW_BASE}/marcap-{year}.parquet"
    print(f"download {year}: {url}",flush=True)
    req=urllib.request.Request(url,headers={"User-Agent":"dante-backtest/1.0"})
    with urllib.request.urlopen(req,timeout=180) as src,open(tmp,"wb") as dst:
        while True:
            chunk=src.read(1024*1024)
            if not chunk: break
            dst.write(chunk)
    tmp.replace(dest)
    print(f"downloaded {year}: {dest.stat().st_size:,} bytes",flush=True)
    return dest

def date_column(df):
    if "Date" in df.columns: return df
    x=df.reset_index()
    if "Date" in x.columns: return x
    if "index" in x.columns: return x.rename(columns={"index":"Date"})
    raise ValueError(f"Date not found: {list(df.columns)}")

def load_year(path,start,end):
    x=date_column(pd.read_parquet(path))
    required=["Date","Code","Name","Open","High","Low","Close","Volume","Amount","Market"]
    missing=[c for c in required if c not in x.columns]
    if missing: raise ValueError(f"{path}: missing {missing}; got {list(x.columns)}")
    ratio_col=next((c for c in ["ChagesRatio","ChangesRatio","ChangeRatio"] if c in x.columns),None)
    keep=required+([ratio_col] if ratio_col else [])
    x=x[keep].copy()
    x["Date"]=pd.to_datetime(x["Date"],errors="coerce")
    x=x[(x["Date"]>=start)&(x["Date"]<=end)]
    x["Market"]=x["Market"].astype(str).str.upper()
    x=x[x["Market"].isin(["KOSPI","KOSDAQ"])].copy()
    if ratio_col: x=x.rename(columns={ratio_col:"reported_change_pct"})
    else: x["reported_change_pct"]=np.nan
    return x

def add_episode_ids(x,episode_gap_days,name_change_gap_days):
    x=x.sort_values(["Code","Date"]).copy()
    x["Code"]=x["Code"].astype(str).str.replace(r"\.0$","",regex=True).str.zfill(6)
    x["Name"]=x["Name"].astype(str)
    prev_date=x.groupby("Code",sort=False)["Date"].shift(1)
    prev_name=x.groupby("Code",sort=False)["Name"].shift(1)
    gap=(x["Date"]-prev_date).dt.days
    new_episode=(prev_date.isna()|(gap>episode_gap_days)|((gap>name_change_gap_days)&(x["Name"]!=prev_name)))
    x["episode_no"]=new_episode.groupby(x["Code"]).cumsum().astype(int)
    x["series_id"]=x["Code"]+"_E"+x["episode_no"].astype(str).str.zfill(2)
    return x

def add_research_adjusted_close(x,diff_threshold):
    x=x.sort_values(["series_id","Date"]).copy()
    for c in ["Open","High","Low","Close","Volume","Amount","reported_change_pct"]:
        x[c]=pd.to_numeric(x[c],errors="coerce")
    raw_ret=x.groupby("series_id",sort=False)["Close"].pct_change(fill_method=None)
    reported=x["reported_change_pct"]/100.0
    reported_valid=reported.between(-0.40,0.40,inclusive="both")
    mismatch=reported_valid&raw_ret.notna()&((raw_ret-reported).abs()>diff_threshold)
    effective=raw_ret.copy()
    effective[mismatch]=reported[mismatch]
    first=x.groupby("series_id",sort=False).cumcount().eq(0)
    effective[first]=0.0
    effective=effective.replace([np.inf,-np.inf],np.nan).fillna(0.0).clip(lower=-0.95,upper=5.0)
    log_growth=np.log1p(effective)
    cum_log=log_growth.groupby(x["series_id"],sort=False).cumsum()
    first_close=x.groupby("series_id",sort=False)["Close"].transform("first")
    x["adjusted_close"]=first_close*np.exp(cum_log)
    x["adjustment_bridge"]=mismatch
    return x

def build_panel(paths,start,end,episode_gap_days,name_change_gap_days,diff_threshold):
    frames=[]
    for i,path in enumerate(paths,1):
        q=load_year(path,start,end); frames.append(q)
        print(f"read {path.name}: {len(q):,} rows ({i}/{len(paths)})",flush=True)
    x=pd.concat(frames,ignore_index=True)
    x=x.dropna(subset=["Date","Code","Open","High","Low","Close","Volume"])
    x=x[(x["Open"]>0)&(x["High"]>0)&(x["Low"]>0)&(x["Close"]>0)]
    x=x.sort_values(["Code","Date"]).drop_duplicates(["Code","Date"],keep="last")
    x=add_episode_ids(x,episode_gap_days,name_change_gap_days)
    x=add_research_adjusted_close(x,diff_threshold)
    x["exchange"]=x["Market"].map({"KOSPI":"KO","KOSDAQ":"KQ"})
    x=x.rename(columns={"Date":"date","Code":"code","Name":"name","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume","Amount":"amount"})
    keep=["series_id","code","exchange","name","date","open","high","low","close","adjusted_close","volume","amount","reported_change_pct","adjustment_bridge"]
    return x[keep].sort_values(["series_id","date"]).reset_index(drop=True)

def main():
    a=parse_args()
    start=pd.Timestamp(a.start)
    end=pd.Timestamp(a.end) if a.end else pd.Timestamp.today().normalize()
    a.out.mkdir(parents=True,exist_ok=True)
    years=list(range(start.year,end.year+1))
    paths=[download_year(y,a.cache,refresh=(a.refresh_end_year and y==end.year)) for y in years]
    panel=build_panel(paths,start,end,a.episode_gap_days,a.name_change_gap_days,a.corp_action_diff)
    ko=panel[panel["exchange"].eq("KO")].copy()
    kq=panel[panel["exchange"].eq("KQ")].copy()
    ko.to_parquet(a.out/"ko_eod.parquet",index=False)
    kq.to_parquet(a.out/"kq_eod.parquet",index=False)
    episodes=(panel.groupby(["series_id","code"],as_index=False).agg(exchange=("exchange","last"),name=("name","last"),first_date=("date","min"),last_date=("date","max"),rows=("date","size"),adjustment_bridges=("adjustment_bridge","sum")))
    episodes["appears_at_end"]=episodes["last_date"]>=(end-pd.Timedelta(days=10))
    episodes.to_csv(a.out/"universe.csv",index=False,encoding="utf-8-sig")
    print(f"panel rows={len(panel):,} episodes={panel['series_id'].nunique():,} KO rows={len(ko):,} KQ rows={len(kq):,} historical-only episodes={(~episodes['appears_at_end']).sum():,}",flush=True)
    print(f"corporate-action bridges={int(panel['adjustment_bridge'].sum()):,} ({panel['adjustment_bridge'].mean():.4%} of rows)",flush=True)

if __name__=="__main__":
    main()
