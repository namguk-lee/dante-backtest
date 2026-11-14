#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full '공구리' classic structure under BOTH old and cap-aware gates.

As-of market snapshot only: not an event-probability or proprietary signal
replica. Distinguishes base structure, volume anchor, resistance breakout, and
EMA224 hold. Does not treat "base exists" as "new daily breakout".
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from dante_classic_scanner import (feat, load, find_bowl, find_anchor,
                                  find_concrete, classify_one)
from cap_liquidity import describe_liquidity

MODES=("legacy","cap-relative")
BASE_THRESHOLDS=(60,90)
NECESSARY=["bowl2_days","bowl1_days","bowl1_decline_pct",
           "bowl2_core_range_pct","bowl2_core_net_change_pct",
           "below224_days_before_approach"]


def is_full_bowl(b):
    if not b:return False
    return bool(
        b["bowl2_days"]>=60 and
        b["bowl2_days"]>=b["bowl1_days"] and
        b["bowl1_decline_pct"]<=-.20 and
        b["below224_days_before_approach"]>=80 and
        np.isfinite(b["bowl2_core_range_pct"]) and
        b["bowl2_core_range_pct"]<=.35 and
        abs(b["bowl2_core_net_change_pct"])<=.20
    )


def report_episode(g, cutoff):
    g=g.sort_values("date")
    if len(g)<520 or pd.Timestamp(g.date.iloc[-1])!=cutoff:
        return None,0
    z=feat(g)
    bowl=find_bowl(z)
    if not bowl:
        return None,1
    cur=z.iloc[-1]
    full=is_full_bowl(bowl)
    # Only fully verified 60-session bases belong in the "Gongguri" page.
    if not full:
        return None,1
    bdays=int(bowl["bowl2_days"])
    features=dict(
        date=str(cur.date.date()),code=str(cur.code).zfill(6),
        name=str(cur["name"]),exchange=str(cur.exchange),
        series_id=str(cur.series_id),
        base_days=bdays,base_type="B90" if bdays>=90 else "B60",
        bowl1_days=int(bowl["bowl1_days"]),
        bowl2_core_days=int(bowl["bowl2_core_days"]),
        bowl1_decline_pct=100*bowl["bowl1_decline_pct"],
        bowl2_width_pct=100*bowl["bowl2_range_pct"],
        bowl2_core_width_pct=100*bowl["bowl2_core_range_pct"],
        bowl2_core_change_pct=100*bowl["bowl2_core_net_change_pct"],
        ema224_below_prior=int(bowl["below224_days_before_approach"]),
        bowl_started=str(z.date.iloc[bowl["base_start_i"]].date()),
        approach_date=str(z.date.iloc[bowl["approach_i"]].date()),
        current_close=float(cur.close),
        price_vs_ema224_pct=100*(float(cur.ac/cur.ema224)-1),
        **describe_liquidity(cur),
    )
    for mode in MODES:
        anchor=find_anchor(z,bowl,5e9,mode)
        label="legacy" if mode=="legacy" else "cap"
        features[f"{label}_anchor"]=bool(anchor)
        features[f"{label}_anchor_day"]=str(z.date.iloc[anchor["anchor_i"]].date()) if anchor else ""
        features[f"{label}_anchor_volume_multiple"]=float(anchor["anchor_volume_ratio"]) if anchor else np.nan
        features[f"{label}_anchor_ret_pct"]=float(anchor["anchor_ret_pct"]*100) if anchor else np.nan
        features[f"{label}_anchor_hold"]=bool(anchor["anchor_alive_strict"]) if anchor else False
        features[f"{label}_anchor_liquidity_pass"]=(
            bool(z.adaptive_liquidity_pass.iloc[anchor["anchor_i"]])
            if anchor and mode=="cap-relative" else
            bool(z.amt20.iloc[anchor["anchor_i"]]>=5e9) if anchor else False
        )
        conc=find_concrete(z,anchor["anchor_i"]) if anchor else None
        features[f"{label}_concrete_breakout"]=bool(conc and conc.get("concrete_breakout",False))
        features[f"{label}_concrete_hold"]=bool(conc and conc.get("concrete_hold_loose",False))
        # Official-style current classic tier, not a recommendation.
        if anchor:
            rr,_=classify_one(g,5e9,mode)
            features[f"{label}_tier"]=rr["tier"] if rr else "UNKNOWN"
            features[f"{label}_ma_state"]=rr["ma_state"] if rr else "UNKNOWN"
        else:
            features[f"{label}_tier"]="NONE"
            features[f"{label}_ma_state"]="NO_ANCHOR"
    features["comparison"]=(
        "BOTH" if features["legacy_anchor"] and features["cap_anchor"] else
        "CAP_ONLY" if features["cap_anchor"] else
        "LEGACY_ONLY" if features["legacy_anchor"] else "BASE_ONLY"
    )
    features["stage_legacy"]=(
        "CONCRETE_HOLD" if features["legacy_concrete_hold"] else
        "ANCHOR_HELD" if features["legacy_anchor_hold"] else
        "ANCHOR_UNHELD" if features["legacy_anchor"] else "BASE_ONLY"
    )
    features["stage_cap"]=(
        "CONCRETE_HOLD" if features["cap_concrete_hold"] else
        "ANCHOR_HELD" if features["cap_anchor_hold"] else
        "ANCHOR_UNHELD" if features["cap_anchor"] else "BASE_ONLY"
    )
    features["buy_signal"]=False
    return features,1


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--ko",type=Path,required=True)
    p.add_argument("--kq",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    market=pd.concat([load(a.ko,"KO"),load(a.kq,"KQ")],ignore_index=True)
    last_day=pd.Timestamp(market.date.max())
    active=market[market.date.eq(last_day)].series_id.astype(str).unique()
    rows=[];tested=0
    for n,(sid,g) in enumerate(market[market.series_id.astype(str).isin(active)].groupby("series_id",sort=False),1):
        r,eligible=report_episode(g,last_day)
        tested+=eligible
        if r:rows.append(r)
        if n%300==0:print("active",n,"tested",tested,"full_bases",len(rows),flush=True)
    if not rows:
        raise RuntimeError("No verified 60-session gongguri bases. Check source coverage.")
    e=pd.DataFrame(rows).sort_values(
        ["comparison","base_days","ema224_below_prior"],
        ascending=[True,False,False])
    e.to_csv(a.out/"all_completed_gongguri_by_liquidity.csv",index=False,encoding="utf-8-sig")
    e[e.comparison.eq("CAP_ONLY")].to_csv(a.out/"cap_only_full_gongguri.csv",index=False,encoding="utf-8-sig")
    e[e.comparison.eq("BOTH")].to_csv(a.out/"both_full_gongguri.csv",index=False,encoding="utf-8-sig")
    e[e.comparison.eq("LEGACY_ONLY")].to_csv(a.out/"legacy_only_full_gongguri.csv",index=False,encoding="utf-8-sig")
    s=(e.groupby(["base_type","comparison","cap_tier"],dropna=False)
       .agg(rows=("code","size"),stocks=("code","nunique"),
            median_bowl_days=("base_days","median"),
            median_marketcap_billion=("market_cap_won",lambda x:np.nanmedian(x)/1e9),
            median_liquidity_billion=("pre20_amount_won",lambda x:np.nanmedian(x)/1e9),
            anchor_held=("cap_anchor_hold","sum"),
            concrete_held=("cap_concrete_hold","sum")).reset_index())
    s.to_csv(a.out/"gongguri_cap_tier_counts.csv",index=False,encoding="utf-8-sig")
    (a.out/"METHOD.txt").write_text(
       "As-of full-market Gongguri stage comparison: decline >=20%, prior EMA224 below >=80 sessions, "
       "bowl2 at least 60 sessions and >=bowl1, early base width <=35%, core net <=20%. "
       "This models Classic bowl1/2/3, not V4 daily first-EMA224 breakout. "
       "A/B/NEAR tier is classic research only, not official Dante paid formula. "
       "Legacy anchor liquidity=20d mean amount INCLUDING anchor bar >=5bn KRW; "
       "cap anchor liquidity=PREVIOUS 20d mean amount and previous cap 3-tier double threshold. "
       "The stock may show bowl state without a fresh breakout TODAY. "
       "Recent Naver cap is only price * last <=5 sessions listed shares, marked estimate. "
       "Different anchor days can be selected under each gate. No orders, no future data.\n",
       encoding="utf-8")
    print("MARKET_LAST",str(last_day.date()),"ACTIVE",len(active),"TESTED",tested,
          "FULL_BOWL",len(e),flush=True)
    print("BY GATE\n",e.comparison.value_counts().to_string(),flush=True)
    print("BY STAGE\n",e.groupby(["comparison","base_type","stage_cap"]).size().to_string(),flush=True)


if __name__=="__main__":
    main()
