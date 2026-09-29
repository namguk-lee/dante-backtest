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
    p.add_argument("--freshen-with-pykrx",action="store_true",
                   help="Append missing recent KOSPI/KOSDAQ daily bars from pykrx after marcap's latest date.")
    p.add_argument("--max-stale-calendar-days",type=int,default=None,
                   help="Fail when the final panel is older than END by more than this many calendar days.")
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

def _pick_column(df,names):
    for name in names:
        if name in df.columns:
            return name
    return None

def append_recent_pykrx(x,end):
    """Append only dates newer than marcap's last row using pykrx bulk daily OHLCV.

    Historical rows continue to come from marcap. This overlay is intentionally
    narrow so a stale upstream annual parquet cannot silently make a "today"
    scan several sessions old.
    """
    if x.empty:
        return x
    latest=pd.to_datetime(x["Date"],errors="coerce").max()
    if pd.isna(latest) or latest>=end:
        print(f"pykrx freshener: no overlay needed (marcap latest={latest})",flush=True)
        return x
    try:
        from pykrx import stock
    except Exception as exc:
        print(f"pykrx freshener unavailable: {exc}",flush=True)
        return x

    meta=(x.sort_values("Date")
            .drop_duplicates("Code",keep="last")
            .assign(Code=lambda q:q["Code"].astype(str).str.replace(r"\\.0$","",regex=True).str.zfill(6))
            .set_index("Code")[["Name","Market"]])
    frames=[]
    first=(latest+pd.Timedelta(days=1)).normalize()
    for dt in pd.date_range(first,end.normalize(),freq="D"):
        ds=dt.strftime("%Y%m%d")
        day_n=0
        for market in ("KOSPI","KOSDAQ"):
            try:
                q=stock.get_market_ohlcv_by_ticker(ds,market=market,alternative=False)
            except Exception as exc:
                print(f"pykrx {ds} {market} failed: {type(exc).__name__}: {exc}",flush=True)
                continue
            if q is None or q.empty:
                continue
            q=q.copy()
            q.index=q.index.astype(str).str.replace(r"\\.0$","",regex=True).str.zfill(6)
            cols={
                "Open":_pick_column(q,["시가","Open"]),
                "High":_pick_column(q,["고가","High"]),
                "Low":_pick_column(q,["저가","Low"]),
                "Close":_pick_column(q,["종가","Close"]),
                "Volume":_pick_column(q,["거래량","Volume"]),
                "Amount":_pick_column(q,["거래대금","Amount","Value"]),
                "reported_change_pct":_pick_column(q,["등락률","Change","ChangeRate"]),
            }
            required=("Open","High","Low","Close","Volume")
            if any(cols[k] is None for k in required):
                print(f"pykrx {ds} {market}: unexpected columns={list(q.columns)}",flush=True)
                continue
            r=pd.DataFrame(index=q.index)
            r["Date"]=dt
            r["Code"]=r.index
            r["Name"]=r.index.map(meta["Name"]) if len(meta) else r.index
            r["Name"]=r["Name"].fillna(r["Code"])
            for out_col in required:
                r[out_col]=pd.to_numeric(q[cols[out_col]],errors="coerce")
            if cols["Amount"] is not None:
                r["Amount"]=pd.to_numeric(q[cols["Amount"]],errors="coerce")
            else:
                r["Amount"]=r["Close"]*r["Volume"]
            if cols["reported_change_pct"] is not None:
                r["reported_change_pct"]=pd.to_numeric(q[cols["reported_change_pct"]],errors="coerce")
            else:
                r["reported_change_pct"]=np.nan
            r["Market"]=market
            r=r.reset_index(drop=True)
            r=r.dropna(subset=["Open","High","Low","Close","Volume"])
            r=r[(r["Open"]>0)&(r["High"]>0)&(r["Low"]>0)&(r["Close"]>0)]
            if not r.empty:
                frames.append(r)
                day_n+=len(r)
        if day_n:
            print(f"pykrx overlay {dt.date()}: {day_n:,} KOSPI/KOSDAQ rows",flush=True)
    if not frames:
        print(f"pykrx freshener: no rows appended after marcap latest={latest.date()}",flush=True)
        return x
    recent=pd.concat(frames,ignore_index=True)
    # Make schemas compatible even when the annual parquet has extra columns.
    for col in x.columns:
        if col not in recent.columns:
            recent[col]=np.nan
    for col in recent.columns:
        if col not in x.columns:
            x[col]=np.nan
    out=pd.concat([x,recent[x.columns]],ignore_index=True)
    out=out.sort_values(["Code","Date"]).drop_duplicates(["Code","Date"],keep="last")
    print(
        f"pykrx freshener: appended {len(recent):,} rows; "
        f"latest {latest.date()} -> {pd.to_datetime(out['Date']).max().date()}",
        flush=True
    )
    return out

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

def build_panel(paths,start,end,episode_gap_days,name_change_gap_days,diff_threshold,freshen_with_pykrx=False):
    frames=[]
    for i,path in enumerate(paths,1):
        q=load_year(path,start,end); frames.append(q)
        print(f"read {path.name}: {len(q):,} rows ({i}/{len(paths)})",flush=True)
    x=pd.concat(frames,ignore_index=True)
    if freshen_with_pykrx:
        x=append_recent_pykrx(x,end)
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
    panel=build_panel(
        paths,start,end,a.episode_gap_days,a.name_change_gap_days,a.corp_action_diff,
        freshen_with_pykrx=a.freshen_with_pykrx
    )
    latest=panel["date"].max() if not panel.empty else pd.NaT
    print(f"final panel latest={latest.date() if pd.notna(latest) else 'NaT'} requested_end={end.date()}",flush=True)
    if a.max_stale_calendar_days is not None:
        stale_days=(end.normalize()-pd.Timestamp(latest).normalize()).days if pd.notna(latest) else 999999
        if stale_days>a.max_stale_calendar_days:
            raise RuntimeError(
                f"stale market panel: latest={latest.date() if pd.notna(latest) else None}, "
                f"end={end.date()}, stale_calendar_days={stale_days}, "
                f"allowed={a.max_stale_calendar_days}"
            )
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
