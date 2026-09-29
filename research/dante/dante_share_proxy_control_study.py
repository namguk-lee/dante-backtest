#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Matched-market control study for the candidate Dante share 1:1 recovery ratio.

Research-only. Tests whether the candidate rule is selective or merely common.
"""
from __future__ import annotations
import argparse
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

def one_ratio(g,ma_n,analysis_date):
    z=feat(g[g.date<=analysis_date].copy(),ma_n)
    if len(z)<ma_n+30:return None
    if pd.isna(z.iloc[-1].ma):return None
    selected,allg=choose_cross(z,ma_n)
    if selected is None:return {
        "ratio":np.nan,"above_ma":bool(z.iloc[-1].ac>=z.iloc[-1].ma),
        "dist_ma_pct":float((z.iloc[-1].ac/z.iloc[-1].ma-1)*100),
        "num_crosses":0
    }
    return {
        "ratio":selected.get("ratio_cross_ma_current",np.nan),
        "above_ma":bool(z.iloc[-1].ac>=z.iloc[-1].ma),
        "dist_ma_pct":float((z.iloc[-1].ac/z.iloc[-1].ma-1)*100),
        "num_crosses":len(allg)
    }

def main():
    a=parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    panel=pd.concat([load(a.ko,"KO"),load(a.kq,"KQ")],ignore_index=True)
    cases=pd.read_csv(a.cases,dtype={"code":str})
    cases["code"]=cases.code.astype(str).str.zfill(6)
    cases["analysis_date"]=pd.to_datetime(cases.analysis_date)

    result_rows=[]; stock_rows=[]
    # Study each unique date/MA pair once.
    pairs=cases[["analysis_date","share_ma"]].drop_duplicates().sort_values(["analysis_date","share_ma"])
    for _,pair in pairs.iterrows():
        dt=pd.Timestamp(pair.analysis_date); ma_n=int(pair.share_ma)
        snap=panel[panel.date<=dt]
        active_ids=set(snap.loc[snap.date.eq(snap.date[snap.date<=dt].max()),"series_id"].astype(str))
        vals=[]
        for sid,g in snap[snap.series_id.astype(str).isin(active_ids)].groupby("series_id",sort=False):
            r=one_ratio(g,ma_n,dt)
            if r is None:continue
            last=g[g.date<=dt].sort_values("date").iloc[-1]
            vals.append({
                "analysis_date":dt,"share_ma":ma_n,"series_id":sid,
                "code":str(last.code).zfill(6),"name":last.get("name",""),
                **r
            })
        q=pd.DataFrame(vals)
        if q.empty:continue
        eligible=q[q.above_ma & q.ratio.notna()].copy()
        ge1=eligible[eligible.ratio>=1]
        result_rows.append({
            "analysis_date":dt,"share_ma":ma_n,
            "active_n":len(q),"eligible_above_with_ratio_n":len(eligible),
            "ratio_ge1_n":len(ge1),
            "ratio_ge1_pct_of_eligible":len(ge1)/len(eligible)*100 if len(eligible) else np.nan,
            "eligible_ratio_median":eligible.ratio.median() if len(eligible) else np.nan,
            "eligible_ratio_p75":eligible.ratio.quantile(.75) if len(eligible) else np.nan,
            "eligible_ratio_p90":eligible.ratio.quantile(.90) if len(eligible) else np.nan,
        })
        stock_rows.extend(q.to_dict("records"))

    stocks=pd.DataFrame(stock_rows)
    summary=pd.DataFrame(result_rows)
    labeled=[]
    for case in cases.to_dict("records"):
        q=stocks[(stocks.analysis_date.eq(pd.Timestamp(case["analysis_date"])))&
                 (stocks.share_ma.eq(int(case["share_ma"])))&
                 (stocks.code.eq(case["code"]))]
        if q.empty:
            labeled.append({**case,"matched_market_control":False});continue
        r=q.iloc[0]
        universe=stocks[(stocks.analysis_date.eq(pd.Timestamp(case["analysis_date"])))&
                        (stocks.share_ma.eq(int(case["share_ma"])))&
                        stocks.above_ma & stocks.ratio.notna()].copy()
        pct=(universe.ratio<=r.ratio).mean()*100 if len(universe) and pd.notna(r.ratio) else np.nan
        labeled.append({
            **case,"matched_market_control":True,
            "candidate_ratio":r.ratio,"above_ma":r.above_ma,"dist_ma_pct":r.dist_ma_pct,
            "ratio_percentile_among_eligible":pct,
            "eligible_universe_n":len(universe),
            "eligible_ge1_pct":float((universe.ratio>=1).mean()*100) if len(universe) else np.nan,
        })

    summary.to_csv(a.out/"share_proxy_market_summary.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(labeled).to_csv(a.out/"share_proxy_labeled_vs_market.csv",index=False,encoding="utf-8-sig")
    stocks.to_csv(a.out/"share_proxy_market_all.csv",index=False,encoding="utf-8-sig")

    print("=== MARKET SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== LABELED VS MARKET ===")
    print(pd.DataFrame(labeled).to_string(index=False))

if __name__=="__main__":
    main()
