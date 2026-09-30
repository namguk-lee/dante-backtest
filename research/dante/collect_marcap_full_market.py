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
    p.add_argument("--freshen-with-naver",action="store_true",
                   help="Disabled: ambiguous venue data. Use --krx-overlay with verified closed bars.")
    p.add_argument("--krx-overlay",type=Path,
                   help="Verified KRX-only CSV: date,code,open,high,low,close,volume,amount,price_venue,bar_status,source_url,verification_url.")
    p.add_argument("--naver-workers",type=int,default=12,
                   help="Parallel workers for the narrow recent-bar Naver overlay.")
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
    # marcap records the KOSDAQ Global segment separately. It remains part
    # of the KOSDAQ universe; excluding it creates multi-year history gaps.
    x["Market"]=x["Market"].replace({"KOSDAQ GLOBAL":"KOSDAQ"})
    x=x[x["Market"].isin(["KOSPI","KOSDAQ"])].copy()
    if ratio_col: x=x.rename(columns={ratio_col:"reported_change_pct"})
    else: x["reported_change_pct"]=np.nan
    return x

def _to_number(value):
    if value is None:
        return np.nan
    if isinstance(value,(int,float,np.integer,np.floating)):
        return float(value)
    s=str(value).replace(",","").replace("%","").strip()
    if not s or s in {"-","--","N/A","None"}:
        return np.nan
    return pd.to_numeric(s,errors="coerce")

def last_closed_date(now=None):
    """Calendar cutoff in Seoul; holidays still require the freshness check."""
    now=pd.Timestamp.now(tz="Asia/Seoul") if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    now=now.tz_convert("Asia/Seoul")
    cutoff=now.normalize()
    if now.hour*60+now.minute<15*60+40:
        cutoff-=pd.Timedelta(days=1)
    while cutoff.weekday()>=5:
        cutoff-=pd.Timedelta(days=1)
    return cutoff.tz_localize(None)


def append_recent_naver(x,end,workers=12):
    raise RuntimeError("Naver freshness disabled: venue-specific responses disagreed with KRX. Supply --krx-overlay with independently verified CLOSED KRX bars.")


def append_verified_krx(x,path,end,now=None):
    """Accept a reviewed provenance ledger, never infer venue from a URL.

    The caller must verify the supplied OHLCV externally. This validator checks
    its declared contract, chronology and integrity; it does not certify prices.
    """
    from urllib.parse import urlparse
    q=pd.read_csv(path,dtype={"code":str})
    required=["date","code","open","high","low","close","volume","amount",
              "price_venue","bar_status","source_url","verification_url"]
    missing=[c for c in required if c not in q]
    if missing:raise ValueError(f"KRX overlay missing {missing}")
    if q.empty:raise ValueError("KRX overlay is empty")
    q["date"]=pd.to_datetime(q.date,errors="raise")
    q["code"]=q.code.str.zfill(6)
    if not q.code.str.fullmatch(r"[0-9]{6}").all():raise ValueError("invalid overlay code")
    if q.duplicated(["code","date"]).any():raise ValueError("duplicate overlay code/date")
    if not q.price_venue.eq("KRX").all() or not q.bar_status.eq("CLOSED").all():
        raise ValueError("overlay requires CLOSED KRX bars")
    if (q.date!=q.date.dt.normalize()).any():raise ValueError("overlay dates must be session dates")
    if (q.date>last_closed_date(now)).any():raise ValueError("incomplete session in overlay")
    if (q.date>pd.Timestamp(end)).any():raise ValueError("overlay extends past analysis cutoff")
    for c in ("open","high","low","close","volume","amount"):
        q[c]=pd.to_numeric(q[c],errors="raise")
        if not np.isfinite(q[c]).all():raise ValueError(f"nonfinite overlay {c}")
    if not (q[["open","high","low","close"]]>0).all().all():raise ValueError("nonpositive OHLC")
    if (q.volume<=0).any() or (q.amount<=0).any():raise ValueError("overlay requires traded volume/value")
    if ((q.high<q[["open","close","low"]].max(axis=1))|
        (q.low>q[["open","close","high"]].min(axis=1))).any():raise ValueError("invalid OHLC envelope")
    for row in q.itertuples():
        urls=[urlparse(str(row.source_url)),urlparse(str(row.verification_url))]
        if any(u.scheme!="https" or not u.hostname for u in urls):raise ValueError("invalid provenance URL")
        if urls[0].hostname==urls[1].hostname:raise ValueError("independent verification host required")
    latest=pd.to_datetime(x.Date).max()
    if (q.date<=latest).any():raise ValueError("overlay must be newer than reference history")
    meta=x[x.Date.eq(latest)].copy();meta["Code"]=meta.Code.astype(str).str.zfill(6)
    meta=meta.drop_duplicates("Code").set_index("Code")
    if not set(q.code)<=set(meta.index):raise ValueError("overlay contains unknown/inactive code")
    q["Name"]=q.code.map(meta.Name);q["Market"]=q.code.map(meta.Market)
    q["reported_change_pct"]=np.nan
    q["verification_status"]="REVIEWED_OVERLAY"
    for col in ("verification_scope","amount_precision"):
        if col not in q:q[col]="unspecified"
    q=q.rename(columns={"date":"Date","code":"Code","open":"Open","high":"High","low":"Low",
                        "close":"Close","volume":"Volume","amount":"Amount"})
    print(f"reviewed KRX overlay: {len(q)} rows, {q.Code.nunique()} codes; coverage is not full-market",flush=True)
    return pd.concat([x,q],ignore_index=True)

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

def build_panel(paths,start,end,episode_gap_days,name_change_gap_days,diff_threshold,freshen_with_naver=False,naver_workers=12,krx_overlay=None):
    frames=[]
    for i,path in enumerate(paths,1):
        q=load_year(path,start,end); frames.append(q)
        print(f"read {path.name}: {len(q):,} rows ({i}/{len(paths)})",flush=True)
    x=pd.concat(frames,ignore_index=True)
    x["price_venue"]="KRX"
    x["bar_status"]="CLOSED"
    x["source_url"]=RAW_BASE
    x["verification_url"]=""
    x["verification_status"]="REFERENCE_HISTORY"
    x["verification_scope"]="provider reference; not independently checked per bar"
    x["amount_precision"]="provider value"
    if freshen_with_naver:
        x=append_recent_naver(x,end,workers=naver_workers)
    if krx_overlay is not None:
        x=append_verified_krx(x,krx_overlay,end)
    x=x.dropna(subset=["Date","Code","Open","High","Low","Close","Volume"])
    x=x[(x["Open"]>0)&(x["High"]>0)&(x["Low"]>0)&(x["Close"]>0)]
    x=x.sort_values(["Code","Date"]).drop_duplicates(["Code","Date"],keep="last")
    x=add_episode_ids(x,episode_gap_days,name_change_gap_days)
    x=add_research_adjusted_close(x,diff_threshold)
    x["exchange"]=x["Market"].map({"KOSPI":"KO","KOSDAQ":"KQ"})
    x=x.rename(columns={"Date":"date","Code":"code","Name":"name","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume","Amount":"amount"})
    keep=["series_id","code","exchange","name","date","open","high","low","close","adjusted_close","volume","amount","reported_change_pct","adjustment_bridge","price_venue","bar_status","source_url","verification_url","verification_status","verification_scope","amount_precision"]
    return x[keep].sort_values(["series_id","date"]).reset_index(drop=True)

def assert_complete_recent_coverage(panel):
    reference=panel[panel.verification_status.eq("REFERENCE_HISTORY")]
    reference_date=reference.date.max()
    latest=panel.date.max()
    expected=set(reference.loc[reference.date.eq(reference_date),"code"])
    observed=set(panel.loc[panel.date.eq(latest),"code"])
    if pd.notna(reference_date) and latest>reference_date and expected-observed:
        raise RuntimeError(f"partial latest-date coverage: {len(expected & observed)}/{len(expected)} reference-active codes; not a fresh full-market panel")

def main():
    a=parse_args()
    start=pd.Timestamp(a.start)
    end=pd.Timestamp(a.end) if a.end else last_closed_date()
    if end.normalize()>last_closed_date():raise ValueError("END is not a closed KRX session date")
    if a.freshen_with_naver:append_recent_naver(pd.DataFrame(),end)
    a.out.mkdir(parents=True,exist_ok=True)
    years=list(range(start.year,end.year+1))
    paths=[download_year(y,a.cache,refresh=(a.refresh_end_year and y==end.year)) for y in years]
    panel=build_panel(
        paths,start,end,a.episode_gap_days,a.name_change_gap_days,a.corp_action_diff,
        freshen_with_naver=a.freshen_with_naver,
        naver_workers=a.naver_workers,krx_overlay=a.krx_overlay
    )
    latest=panel["date"].max() if not panel.empty else pd.NaT
    print(f"final panel latest={latest.date() if pd.notna(latest) else 'NaT'} requested_end={end.date()}",flush=True)
    if a.max_stale_calendar_days is not None:
        if a.krx_overlay is not None:
            assert_complete_recent_coverage(panel)
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
