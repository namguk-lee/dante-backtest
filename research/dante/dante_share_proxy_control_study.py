#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Matched-market control study for Dante share 1:1 candidate geometries.

Research-only. Tests whether candidate ratios that fit labeled public examples
are actually selective versus same-date KOSPI/KOSDAQ controls.
"""
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd
from dante_share_1to1_study import load, feat, choose_cross

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

def select_long_below(allg,min_days=30):
    q=[g for g in allg if float(g.get("pre_days",0) or 0)>=min_days]
    if not q:return None
    return sorted(q,key=lambda x:pd.Timestamp(x["cross_date"]))[-1]

def one_metrics(g,ma_n,analysis_date):
    z=feat(g[g.date<=analysis_date].copy(),ma_n)
    if len(z)<ma_n+30 or pd.isna(z.iloc[-1].ma):return None
    selected,allg=choose_cross(z,ma_n)
    longsel=select_long_below(allg,30)
    return {
        "above_ma":bool(z.iloc[-1].ac>=z.iloc[-1].ma),
        "dist_ma_pct":float((z.iloc[-1].ac/z.iloc[-1].ma-1)*100),
        "num_crosses":len(allg),
        "latest_ratio":selected.get("ratio_cross_ma_current",np.nan) if selected else np.nan,
        "long_ratio":longsel.get("ratio_cross_ma_current",np.nan) if longsel else np.nan,
        "long_cycle_mean_ratio":longsel.get("cycle_mean_dist_ratio",np.nan) if longsel else np.nan,
        "long_cycle_height_ratio":longsel.get("cycle_height_current_ratio",np.nan) if longsel else np.nan,
        "long_pre_days":longsel.get("pre_days",np.nan) if longsel else np.nan,
        "long_cross_date":longsel.get("cross_date",pd.NaT) if longsel else pd.NaT,
    }

def metric_market_stats(q,metric):
    e=q[q.above_ma & pd.to_numeric(q[metric],errors="coerce").notna()].copy()
    v=pd.to_numeric(e[metric],errors="coerce")
    pos=v[v>0]
    return {
        f"{metric}_eligible_n":len(pos),
        f"{metric}_median":float(pos.median()) if len(pos) else np.nan,
        f"{metric}_within20_pct":float(((pos>=.8)&(pos<=1.2)).mean()*100) if len(pos) else np.nan,
        f"{metric}_within35_pct":float(((pos>=.65)&(pos<=1.35)).mean()*100) if len(pos) else np.nan,
        f"{metric}_ge1_pct":float((pos>=1).mean()*100) if len(pos) else np.nan,
    }

def labeled_structural_values(out_dir):
    p=out_dir/"share_1to1_all_structural_crosses_v3.csv"
    if not p.exists():return {}
    x=pd.read_csv(p,dtype={"code":str})
    x["code"]=x.code.astype(str).str.zfill(6)
    x["analysis_date"]=pd.to_datetime(x.analysis_date)
    vals={}
    for (dt,code,ma),g in x.groupby(["analysis_date","code","share_ma"],dropna=False):
        q=g[pd.to_numeric(g.pre_days,errors="coerce")>=30].copy()
        if q.empty:continue
        q["cross_date"]=pd.to_datetime(q.cross_date)
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

    result_rows=[]; stock_rows=[]
    pairs=cases[["analysis_date","share_ma"]].drop_duplicates().sort_values(["analysis_date","share_ma"])
    for _,pair in pairs.iterrows():
        dt=pd.Timestamp(pair.analysis_date); ma_n=int(pair.share_ma)
        snap=panel[panel.date<=dt]
        last_date=snap.date.max()
        active_ids=set(snap.loc[snap.date.eq(last_date),"series_id"].astype(str))
        vals=[]
        for sid,g in snap[snap.series_id.astype(str).isin(active_ids)].groupby("series_id",sort=False):
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
            rec.update(metric_market_stats(q,m))
        result_rows.append(rec)
        stock_rows.extend(q.to_dict("records"))

    stocks=pd.DataFrame(stock_rows)
    summary=pd.DataFrame(result_rows)
    labeled_geom=labeled_structural_values(a.out)
    labeled=[]
    for case in cases.to_dict("records"):
        dt=pd.Timestamp(case["analysis_date"]); ma_n=int(case["share_ma"]); code=case["code"]
        universe=stocks[(stocks.analysis_date.eq(dt))&(stocks.share_ma.eq(ma_n))&stocks.above_ma].copy()
        base={**case,"matched_market_control":bool(len(universe))}
        gv=labeled_geom.get((dt,code,ma_n),{})
        for m in ("long_ratio","long_cycle_mean_ratio","long_cycle_height_ratio"):
            cand=gv.get(m,np.nan)
            base[m]=cand
            uv=pd.to_numeric(universe[m],errors="coerce")
            uv=uv[uv>0]
            if pd.notna(cand) and cand>0 and len(uv):
                ce=near1(cand); ue=np.abs(np.log(uv))
                base[f"{m}_near1_percentile"]=float((ue>=ce).mean()*100)
                base[f"{m}_market_within35_pct"]=float(((uv>=.65)&(uv<=1.35)).mean()*100)
            else:
                base[f"{m}_near1_percentile"]=np.nan
                base[f"{m}_market_within35_pct"]=np.nan
        base["long_pre_days"]=gv.get("long_pre_days",np.nan)
        base["long_cross_date"]=gv.get("long_cross_date",pd.NaT)
        base["eligible_universe_n"]=len(universe)
        labeled.append(base)

    summary.to_csv(a.out/"share_proxy_market_summary_v2.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(labeled).to_csv(a.out/"share_proxy_labeled_vs_market_v2.csv",index=False,encoding="utf-8-sig")
    stocks.to_csv(a.out/"share_proxy_market_all_v2.csv",index=False,encoding="utf-8-sig")

    print("=== MARKET SUMMARY V2 ===")
    print(summary.to_string(index=False))
    print("\n=== LABELED VS MARKET V2 ===")
    print(pd.DataFrame(labeled).to_string(index=False))

if __name__=="__main__":
    main()
