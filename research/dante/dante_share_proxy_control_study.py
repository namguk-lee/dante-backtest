#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast matched-market control study for Dante share 1:1 candidate geometries.

Research-only. Computes only the three candidate metrics needed for controls.
"""
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd
from dante_share_1to1_study import (
    load, feat, structural_up_crosses, contiguous_runs, prior_down_cross, safe_ratio
)

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--ko",type=Path,required=True)
    p.add_argument("--kq",type=Path,required=True)
    p.add_argument("--cases",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("dante_share_study"))
    return p.parse_args()

def near1(v):
    if pd.isna(v) or v<=0:return np.nan
    return abs(math.log(float(v)))

def lightweight_long_metrics(z,ma_n,min_days=30):
    crosses=structural_up_crosses(z,ma_n)
    chosen=None
    for cross_i,_,_ in reversed(crosses):
        ps,pe,_,_=contiguous_runs(z,cross_i)
        pre_days=pe-ps+1
        if pre_days>=min_days:
            chosen=(cross_i,ps,pe,pre_days)
            break
    if chosen is None:return None
    cross_i,ps,pe,pre_days=chosen
    pre=z.iloc[ps:pe+1]
    ma0=float(z.at[cross_i,"ma"])
    current=float(z.iloc[-1].ac)
    pre_low=float(pre.al.min())
    down=(ma0-pre_low)/ma0 if ma0>0 else np.nan
    up=(current-ma0)/ma0 if ma0>0 else np.nan
    long_ratio=safe_ratio(up,down)

    dc=prior_down_cross(z,cross_i,ma_n)
    start=max(0,cross_i-max(60,ma_n)) if dc is None else dc
    seg=z.iloc[start:].copy()
    seg=seg[pd.notna(seg.ma)]
    up_mean=float(seg.loc[seg.dist>0,"dist"].mean()) if (seg.dist>0).any() else np.nan
    dn_mean=float((-seg.loc[seg.dist<0,"dist"]).mean()) if (seg.dist<0).any() else np.nan
    cycle_mean=safe_ratio(up_mean,dn_mean)

    pre_cycle=z.iloc[start:cross_i]
    neg_depth=float((-pre_cycle.dist).clip(lower=0).max()) if len(pre_cycle) else np.nan
    pos_current=float(max(0,z.iloc[-1].dist))
    cycle_height=safe_ratio(pos_current,neg_depth)

    return {
        "long_ratio":long_ratio,
        "long_cycle_mean_ratio":cycle_mean,
        "long_cycle_height_ratio":cycle_height,
        "long_pre_days":pre_days,
        "long_cross_date":z.at[cross_i,"date"],
    }

def one_metrics(g,ma_n,analysis_date):
    z=feat(g[g.date<=analysis_date].copy(),ma_n)
    if len(z)<ma_n+30 or pd.isna(z.iloc[-1].ma):return None
    m=lightweight_long_metrics(z,ma_n,30)
    return {
        "above_ma":bool(z.iloc[-1].ac>=z.iloc[-1].ma),
        "dist_ma_pct":float((z.iloc[-1].ac/z.iloc[-1].ma-1)*100),
        **(m or {
            "long_ratio":np.nan,"long_cycle_mean_ratio":np.nan,
            "long_cycle_height_ratio":np.nan,"long_pre_days":np.nan,"long_cross_date":pd.NaT
        })
    }

def market_stats(q,metric):
    e=pd.to_numeric(q.loc[q.above_ma,metric],errors="coerce")
    e=e[e>0]
    return {
        f"{metric}_eligible_n":len(e),
        f"{metric}_median":float(e.median()) if len(e) else np.nan,
        f"{metric}_within20_pct":float(((e>=.8)&(e<=1.2)).mean()*100) if len(e) else np.nan,
        f"{metric}_within35_pct":float(((e>=.65)&(e<=1.35)).mean()*100) if len(e) else np.nan,
        f"{metric}_ge1_pct":float((e>=1).mean()*100) if len(e) else np.nan,
    }

def labeled_from_crosses(out_dir):
    p=out_dir/"share_1to1_all_structural_crosses_v3.csv"
    if not p.exists():return {}
    x=pd.read_csv(p,dtype={"code":str})
    x["code"]=x.code.astype(str).str.zfill(6)
    x["analysis_date"]=pd.to_datetime(x.analysis_date)
    x["cross_date"]=pd.to_datetime(x.cross_date)
    vals={}
    for (dt,code,ma),g in x.groupby(["analysis_date","code","share_ma"],dropna=False):
        q=g[pd.to_numeric(g.pre_days,errors="coerce")>=30].copy()
        if q.empty:continue
        r=q.sort_values("cross_date").iloc[-1]
        vals[(pd.Timestamp(dt),str(code).zfill(6),int(ma))]={
            "long_ratio":r.get("ratio_cross_ma_current",np.nan),
            "long_cycle_mean_ratio":r.get("cycle_mean_dist_ratio",np.nan),
            "long_cycle_height_ratio":r.get("cycle_height_current_ratio",np.nan),
            "long_pre_days":r.get("pre_days",np.nan),
            "long_cross_date":r.get("cross_date",pd.NaT),
        }
    return vals

def main():
    a=parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    panel=pd.concat([load(a.ko,"KO"),load(a.kq,"KQ")],ignore_index=True)
    cases=pd.read_csv(a.cases,dtype={"code":str})
    cases["code"]=cases.code.astype(str).str.zfill(6)
    cases["analysis_date"]=pd.to_datetime(cases.analysis_date)
    confirmed=cases[cases.label.eq("confirmed_1to1")].copy()

    stock_rows=[]; summary_rows=[]
    pairs=confirmed[["analysis_date","share_ma"]].drop_duplicates().sort_values(["analysis_date","share_ma"])
    for _,pair in pairs.iterrows():
        dt=pd.Timestamp(pair.analysis_date); ma_n=int(pair.share_ma)
        snap=panel[panel.date<=dt]
        last_date=snap.date.max()
        active=set(snap.loc[snap.date.eq(last_date),"series_id"].astype(str))
        vals=[]
        for sid,g in snap[snap.series_id.astype(str).isin(active)].groupby("series_id",sort=False):
            r=one_metrics(g,ma_n,dt)
            if r is None:continue
            last=g[g.date<=dt].sort_values("date").iloc[-1]
            vals.append({
                "analysis_date":dt,"share_ma":ma_n,"series_id":sid,
                "code":str(last.code).zfill(6),"name":last.get("name",""),**r
            })
        q=pd.DataFrame(vals)
        if q.empty:continue
        rec={"analysis_date":dt,"share_ma":ma_n,"active_n":len(q)}
        for m in ("long_ratio","long_cycle_mean_ratio","long_cycle_height_ratio"):
            rec.update(market_stats(q,m))
        summary_rows.append(rec)
        stock_rows.extend(q.to_dict("records"))

    stocks=pd.DataFrame(stock_rows)
    summary=pd.DataFrame(summary_rows)
    labeled_geom=labeled_from_crosses(a.out)
    labeled=[]
    for case in confirmed.to_dict("records"):
        dt=pd.Timestamp(case["analysis_date"]); ma_n=int(case["share_ma"]); code=case["code"]
        universe=stocks[(stocks.analysis_date.eq(dt))&(stocks.share_ma.eq(ma_n))&stocks.above_ma].copy()
        gv=labeled_geom.get((dt,code,ma_n),{})
        rec={**case,"eligible_universe_n":len(universe)}
        for m in ("long_ratio","long_cycle_mean_ratio","long_cycle_height_ratio"):
            cand=gv.get(m,np.nan); rec[m]=cand
            uv=pd.to_numeric(universe[m],errors="coerce"); uv=uv[uv>0]
            if pd.notna(cand) and cand>0 and len(uv):
                ce=near1(cand); ue=np.abs(np.log(uv))
                rec[f"{m}_near1_percentile"]=float((ue>=ce).mean()*100)
                rec[f"{m}_market_within35_pct"]=float(((uv>=.65)&(uv<=1.35)).mean()*100)
                rec[f"{m}_market_median"]=float(uv.median())
            else:
                rec[f"{m}_near1_percentile"]=np.nan
                rec[f"{m}_market_within35_pct"]=np.nan
                rec[f"{m}_market_median"]=np.nan
        rec["long_pre_days"]=gv.get("long_pre_days",np.nan)
        rec["long_cross_date"]=gv.get("long_cross_date",pd.NaT)
        labeled.append(rec)

    summary.to_csv(a.out/"share_proxy_market_summary_v3.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(labeled).to_csv(a.out/"share_proxy_labeled_vs_market_v3.csv",index=False,encoding="utf-8-sig")
    stocks.to_csv(a.out/"share_proxy_market_all_v3.csv",index=False,encoding="utf-8-sig")
    print("=== MARKET SUMMARY V3 ===")
    print(summary.to_string(index=False))
    print("\n=== LABELED VS MARKET V3 ===")
    print(pd.DataFrame(labeled).to_string(index=False))

if __name__=="__main__":
    main()
