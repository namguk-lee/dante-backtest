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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
import urllib.request
import numpy as np
import pandas as pd
import requests

RAW_BASE = "https://raw.githubusercontent.com/FinanceData/marcap/master/data"

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--start",default="2016-01-01")
    p.add_argument("--end",default=None)
    p.add_argument("--out",type=Path,default=Path("krx_data"))
    p.add_argument("--cache",type=Path,default=Path("marcap_cache"))
    p.add_argument("--refresh-end-year",action="store_true")
    p.add_argument("--freshen-with-naver",action="store_true",
                   help="Append missing recent bars from Naver Finance after marcap's latest date.")
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

def append_recent_naver(x,end,workers=12):
    """Append only bars newer than marcap's last date from Naver Finance.

    This is a narrow freshness overlay, not a replacement for marcap history.
    We query only symbols active on marcap's latest session and retain rows
    strictly newer than that session.
    """
    if x.empty:
        return x
    latest=pd.to_datetime(x["Date"],errors="coerce").max()
    if pd.isna(latest) or latest>=end:
        print(f"naver freshener: no overlay needed (marcap latest={latest})",flush=True)
        return x

    meta=(x.sort_values("Date")
            .assign(Code=lambda q:q["Code"].astype(str).str.replace(r"\.0$","",regex=True).str.zfill(6)))
    active=meta[meta["Date"].eq(latest)].drop_duplicates("Code",keep="last")
    info=active.set_index("Code")[["Name","Market"]].to_dict("index")
    codes=sorted(info)
    if not codes:
        print("naver freshener: no active codes on latest marcap date",flush=True)
        return x

    url_tpl="https://m.stock.naver.com/api/stock/{code}/price?pageSize=10&page=1"
    headers={
        "User-Agent":"Mozilla/5.0 (compatible; dante-backtest/1.0)",
        "Referer":"https://m.stock.naver.com/"
    }

    def fetch_one(code):
        last_exc=None
        for attempt in range(3):
            try:
                r=requests.get(url_tpl.format(code=code),headers=headers,timeout=12)
                if r.status_code==429:
                    time.sleep(0.6*(attempt+1))
                    continue
                r.raise_for_status()
                data=r.json()
                if isinstance(data,dict):
                    data=(data.get("priceInfos") or data.get("data") or data.get("result") or [])
                if not isinstance(data,list):
                    return code,[],f"unexpected_json:{type(data).__name__}"
                rows=[]
                for bar in data:
                    if not isinstance(bar,dict):
                        continue
                    dt=pd.to_datetime(bar.get("localTradedAt") or bar.get("localDate"),errors="coerce")
                    if pd.isna(dt) or dt<=latest or dt>end:
                        continue
                    o=_to_number(bar.get("openPrice"))
                    h=_to_number(bar.get("highPrice"))
                    l=_to_number(bar.get("lowPrice"))
                    cl=_to_number(bar.get("closePrice"))
                    v=_to_number(bar.get("accumulatedTradingVolume"))
                    if not all(pd.notna(vv) and float(vv)>0 for vv in (o,h,l,cl)):
                        continue
                    if pd.isna(v) or v<0:
                        v=0.0
                    amount=_to_number(bar.get("accumulatedTradingValue"))
                    if pd.isna(amount):
                        amount=float(cl)*float(v)
                    change=_to_number(bar.get("fluctuationsRatio"))
                    m=info.get(code,{})
                    rows.append({
                        "Date":pd.Timestamp(dt).normalize(),
                        "Code":code,
                        "Name":m.get("Name",code),
                        "Open":o,"High":h,"Low":l,"Close":cl,
                        "Volume":v,"Amount":amount,
                        "Market":m.get("Market"),
                        "reported_change_pct":change,
                    })
                return code,rows,None
            except Exception as exc:
                last_exc=exc
                time.sleep(0.25*(attempt+1))
        return code,[],f"{type(last_exc).__name__}:{last_exc}" if last_exc else "unknown"

    rows=[]; errors=[]; completed=0
    with ThreadPoolExecutor(max_workers=max(1,int(workers))) as pool:
        futs={pool.submit(fetch_one,code):code for code in codes}
        for fut in as_completed(futs):
            code,bars,err=fut.result()
            completed+=1
            if bars:
                rows.extend(bars)
            if err:
                errors.append((code,err))
            if completed%500==0:
                print(f"naver freshener progress: {completed}/{len(codes)} codes, rows={len(rows):,}, errors={len(errors)}",flush=True)

    if errors:
        print(f"naver freshener errors={len(errors)}; samples={errors[:5]}",flush=True)
    if not rows:
        print(f"naver freshener: no rows appended after marcap latest={latest.date()}",flush=True)
        return x

    recent=pd.DataFrame(rows)
    recent=recent.drop_duplicates(["Code","Date"],keep="last")
    counts=recent.groupby("Date").size().sort_index()
    print("naver overlay dates: "+", ".join(f"{d.date()}={int(n):,}" for d,n in counts.items()),flush=True)

    for col in x.columns:
        if col not in recent.columns:
            recent[col]=np.nan
    for col in recent.columns:
        if col not in x.columns:
            x[col]=np.nan
    out=pd.concat([x,recent[x.columns]],ignore_index=True)
    out=out.sort_values(["Code","Date"]).drop_duplicates(["Code","Date"],keep="last")
    print(
        f"naver freshener: appended {len(recent):,} rows; "
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

def build_panel(paths,start,end,episode_gap_days,name_change_gap_days,diff_threshold,freshen_with_naver=False,naver_workers=12):
    frames=[]
    for i,path in enumerate(paths,1):
        q=load_year(path,start,end); frames.append(q)
        print(f"read {path.name}: {len(q):,} rows ({i}/{len(paths)})",flush=True)
    x=pd.concat(frames,ignore_index=True)
    if freshen_with_naver:
        x=append_recent_naver(x,end,workers=naver_workers)
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
        freshen_with_naver=a.freshen_with_naver,
        naver_workers=a.naver_workers
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
