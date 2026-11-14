#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast operational Gongguri page derived from the ALREADY RUN classic scans.

No extra market requests: post-process the identical-date legacy and cap
Classic CSV outputs. This is stricter than merely a strong volume candle and
does not confuse 'old existing bowl' with a new breakout. Every source/quality
warning survives in the exported rows.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd

REQUIRED={
    "date","code","name","bowl2_days","bowl1_days",
    "bowl1_decline_pct","bowl2_core_range_pct","bowl2_core_net_change_pct",
    "below224_days_before_approach","anchor_date","anchor_alive_strict",
    "concrete_breakout","concrete_hold_loose","tier","market_cap_quality",
    "cap_tier","anchor_liquidity_relative_pct",
}

def load_validated(path):
    try:
        x=pd.read_csv(path,dtype={"code":str})
    except pd.errors.EmptyDataError:
        x=pd.DataFrame(columns=sorted(REQUIRED))
    if x.empty:
        if "date" not in x:x["date"]=pd.Series(dtype="datetime64[ns]")
        if "anchor_date" not in x:x["anchor_date"]=pd.Series(dtype="datetime64[ns]")
        return x
    missing=REQUIRED-set(x.columns)
    if missing:raise RuntimeError(f"{path}: missing Classic columns {sorted(missing)}")
    x["date"]=pd.to_datetime(x.date)
    x["anchor_date"]=pd.to_datetime(x.anchor_date)
    x["code"]=x.code.str.zfill(6)
    if x.date.nunique()!=1:
        raise RuntimeError(f"{path}: multiple as-of dates")
    return x

def strict_full_bowl(x):
    return (
       x.bowl2_days.ge(60)&
       x.bowl2_days.ge(x.bowl1_days)&
       x.bowl1_decline_pct.le(-20)&
       x.below224_days_before_approach.ge(80)&
       x.bowl2_core_range_pct.le(35)&
       x.bowl2_core_net_change_pct.abs().le(20)
    ).fillna(False)

def state(d):
    if not bool(d.anchor_alive_strict):
        return "ANCHOR_INVALIDATED"
    if bool(d.concrete_hold_loose):
        return "CONCRETE_HOLD"
    if bool(d.concrete_breakout):
        return "BREAKOUT_NOT_HELD"
    return "ANCHOR_HELD_NOT_BROKEN"

def process(legacy,cap):
    a=load_validated(legacy);b=load_validated(cap)
    dates=set(a.date.dt.strftime("%Y-%m-%d"))|set(b.date.dt.strftime("%Y-%m-%d"))
    if not dates:
        return pd.DataFrame(),pd.DataFrame()
    if len(dates)!=1:raise RuntimeError(f"Legacy/Cap snapshot dates do not match: {dates}")
    a=a[strict_full_bowl(a)].copy()
    b=b[strict_full_bowl(b)].copy()
    out=[]
    for mode,d in (("LEGACY_50E",a),("CAP_RELATIVE",b)):
        for _,r in d.iterrows():
            r=r.to_dict()
            name=str(r.get("name",""))
            preferred=bool(re.search(r"우(?:[0-9]*[BC])?$",name))
            out.append(dict(
                date=str(pd.Timestamp(r["date"]).date()),
                code=str(r["code"]),name=name,exchange=r.get("exchange",""),
                liquidity_gate=mode,
                base_type="B90" if r["bowl2_days"]>=90 else "B60",
                bowl_2_days=int(r["bowl2_days"]),
                bowl_1_days=int(r["bowl1_days"]),
                core_base_width_pct=float(r["bowl2_core_range_pct"]),
                prior_ema224_below_days=int(r["below224_days_before_approach"]),
                decline_pct=float(r["bowl1_decline_pct"]),
                anchor_date=str(pd.Timestamp(r["anchor_date"]).date()),
                anchor_volume_ratio=float(r.get("anchor_volume_ratio",np.nan)),
                anchor_alive_strict=bool(r["anchor_alive_strict"]),
                resistance_broken=bool(r["concrete_breakout"]),
                resistance_supported=bool(r["concrete_hold_loose"]),
                stage=state(pd.Series(r)),
                classic_tier=str(r["tier"]),
                anchor_is_new_on_analysis_date=bool(r["anchor_date"]==r["date"]),
                latest_marketcap_quality=str(r["market_cap_quality"]),
                latest_marketcap_tier=str(r["cap_tier"]),
                anchor_relative_liquidity_pct=float(r["anchor_liquidity_relative_pct"])
                  if pd.notna(r["anchor_liquidity_relative_pct"]) else np.nan,
                liquidity_and_price_warning=(
                  "CURRENT_CAP_UNAVAILABLE" if pd.isna(r.get("market_cap_won"))
                  else "VERIFY_ORDERBOOK"),
                preferred_share_check=preferred,
                no_automatic_buy=True,
            ))
    d=pd.DataFrame(out)
    if d.empty:return d,pd.DataFrame()
    idcounts=d.groupby("code").liquidity_gate.agg(set).to_dict()
    d["comparison"]=d.code.map(lambda c:
        "BOTH" if len(idcounts[c])==2 else
        "CAP_ONLY" if "CAP_RELATIVE" in idcounts[c] else "LEGACY_ONLY")
    # A single code may appear in both modes with different anchor dates.
    summary=(d.groupby(["liquidity_gate","base_type","comparison","stage"],dropna=False)
              .agg(candidate_rows=("code","size"),
                   unique_codes=("code","nunique"),
                   newly_fired_anchor=("anchor_is_new_on_analysis_date","sum"),
                   missing_current_cap=("liquidity_and_price_warning",
                                      lambda s:int(s.eq("CURRENT_CAP_UNAVAILABLE").sum())),
                   preferred_share_check=("preferred_share_check","sum"))
              .reset_index())
    return d,summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--legacy",type=Path,required=True)
    p.add_argument("--cap",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    d,s=process(a.legacy,a.cap)
    cols=["date","code","name","exchange","liquidity_gate","base_type",
          "bowl_2_days","bowl_1_days","core_base_width_pct",
          "prior_ema224_below_days","decline_pct","anchor_date",
          "anchor_volume_ratio","anchor_alive_strict","resistance_broken",
          "resistance_supported","stage","classic_tier","anchor_is_new_on_analysis_date",
          "latest_marketcap_quality","latest_marketcap_tier",
          "anchor_relative_liquidity_pct","liquidity_and_price_warning",
          "preferred_share_check","no_automatic_buy","comparison"]
    if d.empty:
        d=pd.DataFrame(columns=cols)
    d.to_csv(a.out/"verified_gongguri_60_90.csv",index=False,encoding="utf-8-sig")
    if s.empty:
        s=pd.DataFrame(columns=["liquidity_gate","base_type","comparison","stage",
                                "candidate_rows","unique_codes","newly_fired_anchor",
                                "missing_current_cap","preferred_share_check"])
    s.to_csv(a.out/"verified_gongguri_gate_summary.csv",index=False,encoding="utf-8-sig")
    d[d.comparison.eq("CAP_ONLY")].to_csv(
        a.out/"newly_included_cap_only_full_bases.csv",index=False,encoding="utf-8-sig")
    (a.out/"README.txt").write_text(
      "Only Classic B60/B90 with all Gongguri structure rules are present: "
      "Bowl2 duration >=60 and >=Bowl1, decline <=-20%, >=80 below224, "
      "core10-90% close width <=35%, net change <=20%. "
      "Volume/anchor/CONCRETE are separate states. The 'newly included' cap "
      "column means inclusion by liquidity gate, NOT today newly formed "
      "or a buy signal. Gate results may share ticker but DIFFERENT anchor dates. "
      "Latest cap may be absent while historical anchor cap was verified: "
      "declare CURRENT_CAP_UNAVAILABLE and never invent current cap. "
      "Preferred stocks have special liquidity risk. "
      "This run is a point-in-time snapshot and does not establish future returns.\n",
      encoding="utf-8")
    print("COMPLETE_BASES",len(d),"UNIQUE",d.code.nunique(),flush=True)
    print("STAGES\n"+s.to_string(index=False),flush=True)

if __name__=="__main__":main()
