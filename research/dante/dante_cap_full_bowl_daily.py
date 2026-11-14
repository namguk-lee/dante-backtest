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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from dante_classic_scanner import (feat, load, find_bowl, find_anchor,
                                  find_concrete, classify_one, render_chart)
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



def draw_bowl_avwap(z, rec, path):
    """As-of daily candlestick with thick *daily* base-anchored HLC3 VWAP.

    The anchor is the actual BOWL2 start, never a model-optimized offset.
    This is NOT intraday/session VWAP and not a paid Dante proprietary line.
    """
    n=len(z)
    base_i=int(rec["_idx_base_start"])
    approach_i=int(rec["_idx_approach"])
    anchor_i=int(rec["_idx_anchor"])
    left=max(0,min(base_i,n-210))
    right=n
    t=z.iloc[left:right].copy().reset_index(drop=True)
    scale=float(z.ac.iloc[-1]/z.close.iloc[-1])
    if not np.isfinite(scale) or scale<=0:raise RuntimeError("Bad adjusted/raw price scale")
    x=np.arange(len(t))
    fig,(ax,av)=plt.subplots(2,1,figsize=(16.3,9.2),sharex=True,
        gridspec_kw={"height_ratios":[4,1]},facecolor="white")
    for j,r in t.iterrows():
        up=r.ac>=r.ao
        color="#d45658" if up else "#357ab1"
        ax.vlines(j,r.al/scale,r.ah/scale,color=color,linewidth=.8)
        width=.62
        body0=min(r.ao,r.ac)/scale
        height=max(abs(r.ac-r.ao)/scale,.4)
        ax.add_patch(Rectangle((j-width/2,body0),width,height,
                       facecolor=color,edgecolor=color,linewidth=.4))
        av.bar(j,r.volume/1e6,width=.7,color=color,alpha=.85)
    for p,color,lw in ((112,"#2a9e9c",1.5),(224,"#8b55aa",2.4),
                       (448,"#66778b",1.5)):
        ax.plot(x,t[f"ema{p}"]/scale,label=f"EMA{p}",linewidth=lw,color=color)
    # Cumulated daily HLC3 weighted by actual KRX-listed trading volume from
    # the base's true first trading day. Never include earlier or future bars.
    prices=(z.ah+z.al+z.ac)/3
    v=z.volume.astype(float).clip(lower=0)
    valid=prices.iloc[base_i:] * v.iloc[base_i:]
    cumv=v.iloc[base_i:].cumsum().replace(0,np.nan)
    anchored=pd.Series(np.nan,index=z.index,dtype=float)
    anchored.iloc[base_i:]=(valid.cumsum()/cumv).to_numpy()
    ax.plot(x,anchored.iloc[left:right].to_numpy(float)/scale,
            label="BOWL2 DAILY AVWAP (HLC3 x volume)",color="#f59d2d",linewidth=3.4)
    for lab,idx,color in (("Base start",base_i,"#40947a"),
                          ("EMA224 area",approach_i,"#a77b9a"),
                          ("Anchor",anchor_i,"#c98127")):
        loc=idx-left
        if 0<=loc<len(t):
            ax.axvline(loc,color=color,linestyle="--",alpha=.6,linewidth=1.0,label=lab)
    if np.isfinite(rec.get("anchor_open",np.nan)):
        ax.axhline(float(rec["anchor_open"]),color="#b95353",linewidth=1.2,
                   linestyle=":",alpha=.75,label="Anchor open support")
    if np.isfinite(rec.get("concrete_resistance",np.nan)):
        ax.axhline(float(rec["concrete_resistance"]),color="#5e7d92",
                   linestyle=":",linewidth=1.0,alpha=.7,label="Concrete resistance")
    ax.set_title(f'{rec["code"]}  Full Gongguri B{rec["bowl2_days"]} | '
                 f'cap-normalized anchor research',fontsize=15)
    ax.set_ylabel("KRW price")
    ax.grid(alpha=.16);av.grid(alpha=.12)
    ax.legend(loc="upper left",fontsize=8,ncol=3)
    av.set_ylabel("Million shares")
    ticks=list(range(0,len(t),max(1,len(t)//9)))
    if ticks[-1]!=len(t)-1:ticks.append(len(t)-1)
    av.set_xticks(ticks)
    av.set_xticklabels([str(t.date.iloc[k].date()) for k in ticks],
                       rotation=25,ha="right",fontsize=8)
    fig.text(.02,.015,
        "Research only. True DAILY HLC3*volume AVWAP from completed Bowl2 start; "
        "not intraday VWAP. No verified proprietary Dante formula or trade signal.",
        fontsize=9,color="#45536a")
    fig.tight_layout(rect=(0,.03,1,.96))
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=135)
    plt.close(fig)
    return path


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
        features[f"{label}_anchor_cap_tier"]=str(z.at[anchor["anchor_i"],"cap_tier"]) if anchor else "NO_ANCHOR"
        features[f"{label}_anchor_cap_quality"]=str(z.at[anchor["anchor_i"],"market_cap_quality"]) if anchor else "NO_ANCHOR"
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
            features[f"{label}_classic_tier"]=rr["tier"] if rr else "UNKNOWN"
            features[f"{label}_ma_state"]=rr["ma_state"] if rr else "UNKNOWN"
        else:
            features[f"{label}_classic_tier"]="NONE"
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
    p.add_argument("--max-charts",type=int,default=8)
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
    # Actual raw-price/EMA/bar charts for the strict completed bowls, separate
    # from mere technical strong candles and invalidated anchor examples.
    candidates=e[e.cap_anchor].copy()
    order={"CONCRETE_HOLD":0,"ANCHOR_HELD":1,"ANCHOR_UNHELD":2}
    candidates["_s"]=candidates.stage_cap.map(order).fillna(9)
    candidates["_g"]=candidates.comparison.map({"CAP_ONLY":0,"BOTH":1}).fillna(9)
    chosen=candidates.sort_values(["_s","_g","base_days"],ascending=[True,True,False]).head(a.max_charts)
    if len(chosen):
        groups={
            str(sid):g for sid,g in market[market.series_id.astype(str).isin(
                set(chosen.series_id.astype(str)))].groupby("series_id",sort=False)
        }
        for _,t in chosen.iterrows():
            try:
                g=groups[str(t.series_id)]
                rec,z=classify_one(g,5e9,"cap-relative")
                if rec is not None:
                    render_chart(z,rec,a.out/"charts")
                    draw_bowl_avwap(z,rec,a.out/"charts"/f"{rec['code']}_B{int(rec['bowl2_days'])}_DAILY_AVWAP.png")
            except Exception as exc:
                print("CHART_FAILED",t.code,repr(exc),flush=True)
    print("MARKET_LAST",str(last_day.date()),"ACTIVE",len(active),"TESTED",tested,
          "FULL_BOWL",len(e),flush=True)
    print("BY GATE\n",e.comparison.value_counts().to_string(),flush=True)
    print("BY STAGE\n",e.groupby(["comparison","base_type","stage_cap"]).size().to_string(),flush=True)


if __name__=="__main__":
    main()
