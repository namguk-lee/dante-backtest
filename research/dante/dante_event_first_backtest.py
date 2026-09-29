#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Historical validation for the event-first Dante hypothesis.

Question:
Does filtering the universe by a prior strong price/volume event, then waiting
for EMA224 recovery and a lower-volume pullback, improve forward returns?

Transparent approximation only; no proprietary indicators are reproduced.
"""
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--ko",type=Path,required=True); p.add_argument("--kq",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("event_backtest"))
    p.add_argument("--start",default="2018-01-01")
    p.add_argument("--cost-bps",type=float,default=50.0)
    return p.parse_args()

def load(path,ex):
    x=pd.read_parquet(path).copy()
    x["date"]=pd.to_datetime(x.date,errors="coerce")
    if "exchange" not in x:x["exchange"]=ex
    if "adjusted_close" not in x:x["adjusted_close"]=x["close"]
    if "amount" not in x:x["amount"]=x["close"]*x["volume"]
    return x.sort_values(["series_id","date"])

def feat(g):
    g=g.sort_values("date").copy()
    f=(g.adjusted_close/g.close).replace([np.inf,-np.inf],np.nan).ffill().bfill().fillna(1.0)
    g["ao"]=g.open*f; g["ah"]=g.high*f; g["al"]=g.low*f; g["ac"]=g.adjusted_close
    for n in (5,15,33,56,112,224,448):g[f"ema{n}"]=g.ac.ewm(span=n,adjust=False,min_periods=n).mean()
    g["v20"]=g.volume.rolling(20,min_periods=20).mean()
    g["amt20"]=g.amount.rolling(20,min_periods=20).mean()
    g["vr"]=g.volume/g.v20
    g["ret1"]=g.ac.pct_change(fill_method=None)
    g["body"]=(g.ac-g.ao)/g.ao
    rng=(g.ah-g.al).replace(0,np.nan)
    g["close_pos"]=(g.ac-g.al)/rng
    g["upper_wick"]=(g.ah-np.maximum(g.ao,g.ac))/rng
    g["cross224"]=(g.ac>g.ema224)&(g.ac.shift(1)<=g.ema224.shift(1))
    g["below80"]=(g.ac<g.ema224).shift(1).rolling(80,min_periods=80).sum()
    g["prior5_high"]=g.ah.shift(1).rolling(5,min_periods=5).max()
    g["prior20_high"]=g.ah.shift(1).rolling(20,min_periods=20).max()
    g["event_breakout20"]=(g.ac>g.prior20_high)
    g["event_breakout_strength"]=g.ac/g.prior20_high-1
    g["ema112_slope20"]=g.ema112/g.ema112.shift(20)-1
    g["ema224_slope20"]=g.ema224/g.ema224.shift(20)-1
    return g.reset_index(drop=True)

def split(d):
    y=pd.Timestamp(d).year
    return "TRAIN" if y<=2021 else "VALID" if y<=2024 else "TEST"


def simulate_path(z,entry_i,entry,stop,target,horizon,cost):
    """Conservative daily-bar simulation: if stop and target touch same day, stop wins."""
    p=f"path{horizon}_"
    out={
        p+"valid":False,p+"full_horizon":False,p+"outcome":"INVALID",
        p+"days":np.nan,p+"net_return":np.nan,p+"r_multiple":np.nan,
        p+"mfe":np.nan,p+"mae":np.nan,
    }
    if not all(math.isfinite(v) for v in [entry,stop,target]) or not(stop<entry<target):
        return out
    end_i=entry_i+horizon
    full=end_i<len(z)
    last_i=min(end_i,len(z)-1)
    if last_i<entry_i:return out
    path=z.iloc[entry_i:last_i+1]
    out[p+"valid"]=True
    out[p+"full_horizon"]=full
    out[p+"mfe"]=float(path["ah"].max()/entry-1)
    out[p+"mae"]=float(path["al"].min()/entry-1)
    outcome="TIMEOUT" if full else "OPEN_INCOMPLETE"
    exit_price=float(z.at[last_i,"ac"])
    days=int(last_i-entry_i)
    for j in range(entry_i,last_i+1):
        stop_hit=bool(float(z.at[j,"al"])<=stop)
        target_hit=bool(float(z.at[j,"ah"])>=target)
        if stop_hit and target_hit:
            outcome="STOP_SAME_DAY"
            exit_price=stop
            days=int(j-entry_i)
            break
        if stop_hit:
            outcome="STOP"
            exit_price=stop
            days=int(j-entry_i)
            break
        if target_hit:
            outcome="TARGET"
            exit_price=target
            days=int(j-entry_i)
            break
    net_ret=float(exit_price/entry-1-cost)
    risk_pct=float((entry-stop)/entry)
    out[p+"outcome"]=outcome
    out[p+"days"]=days
    out[p+"net_return"]=net_ret
    out[p+"r_multiple"]=float(net_ret/risk_pct) if risk_pct>0 else np.nan
    return out

def add_event(rows,z,kind,signal_i,event_i,cost,context=None):
    entry_i=signal_i+1
    if entry_i>=len(z):return
    entry=float(z.at[entry_i,"ao"])
    if not math.isfinite(entry) or entry<=0:return
    rec={"series_id":z.at[signal_i,"series_id"],"exchange":z.at[signal_i,"exchange"],
         "kind":kind,"event_date":z.at[event_i,"date"],"signal_date":z.at[signal_i,"date"],
         "entry_date":z.at[entry_i,"date"],"entry":entry,"split":split(z.at[signal_i,"date"]),
         "event_quality":"CLEAN_BREAKOUT" if (z.at[event_i,"close_pos"]>=.65 and z.at[event_i,"upper_wick"]<=.35) else "WICK_ENERGY",
         "event_breakout20":bool(z.at[event_i,"event_breakout20"]) if pd.notna(z.at[event_i,"event_breakout20"]) else False,
         "event_breakout_strength":float(z.at[event_i,"event_breakout_strength"]) if pd.notna(z.at[event_i,"event_breakout_strength"]) else np.nan,
         "event_volume_ratio":float(z.at[event_i,"vr"]) if pd.notna(z.at[event_i,"vr"]) else np.nan,
         "event_return":float(z.at[event_i,"ret1"]) if pd.notna(z.at[event_i,"ret1"]) else np.nan,
         "event_body":float(z.at[event_i,"body"]) if pd.notna(z.at[event_i,"body"]) else np.nan,
         "signal_ema224_slope20":float(z.at[signal_i,"ema224_slope20"]) if pd.notna(z.at[signal_i,"ema224_slope20"]) else np.nan,
         "signal_gap112_224":float(abs(z.at[signal_i,"ema224"]-z.at[signal_i,"ema112"])/z.at[signal_i,"ema224"]) if pd.notna(z.at[signal_i,"ema224"]) and z.at[signal_i,"ema224"]>0 and pd.notna(z.at[signal_i,"ema112"]) else np.nan,
         "signal_headroom448":float(z.at[signal_i,"ema448"]/z.at[signal_i,"ac"]-1) if pd.notna(z.at[signal_i,"ema448"]) and z.at[signal_i,"ac"]>0 else np.nan}
    if context:
        rec.update(context)
    if kind in ("E3_CLASSIC_PULLBACK","E4_REACCELERATION") and context:
        stop=float(context.get("anchor_price_adj",np.nan))
        target=float(context.get("target448_signal",np.nan))
        rec.update(simulate_path(z,entry_i,entry,stop,target,20,cost))
        rec.update(simulate_path(z,entry_i,entry,stop,target,60,cost))
    for h in (20,60):
        j=entry_i+h
        if j<len(z):
            gross=float(z.at[j,"ac"]/entry-1)
            rec[f"ret{h}"]=gross-cost
        else:rec[f"ret{h}"]=np.nan
    rows.append(rec)

def scan_one(g,start,cost):
    z=feat(g)
    rows=[]; cooldown={"E0_EVENT":-999,"E1_BOWL_EVENT":-999,"E2_224_RECOVERY":-999,"E3_CLASSIC_PULLBACK":-999,"E4_REACCELERATION":-999}
    start_i=max(500,int(z.index[z.date>=start][0]) if (z.date>=start).any() else len(z))
    strong_mask=(
        (z.ret1>=.05) & (z.body>=.035) & (z.vr>=2) & (z.amt20>=5_000_000_000)
    ).fillna(False).to_numpy()
    event_idx=np.flatnonzero(strong_mask)
    event_idx=event_idx[(event_idx>=start_i)&(event_idx<len(z)-62)]
    for i in event_idx:
        r=z.iloc[i]
        if i-cooldown["E0_EVENT"]>=40:
            add_event(rows,z,"E0_EVENT",i,i,cost); cooldown["E0_EVENT"]=i

        long_below=bool(r.below80>=60)
        reverse=bool(r.ema112<r.ema224<r.ema448) if all(pd.notna(r[x]) for x in ["ema112","ema224","ema448"]) else False
        if not(long_below and reverse):continue
        if i-cooldown["E1_BOWL_EVENT"]>=40:
            add_event(rows,z,"E1_BOWL_EVENT",i,i,cost); cooldown["E1_BOWL_EVENT"]=i

        anchor=float(r.ao)
        # Stage 2: first EMA224 recovery after the event, while anchor remains alive.
        cross_i=None
        for j in range(i,min(i+21,len(z)-1)):
            post=z.iloc[i:j+1]
            if int((post.ac<anchor).sum())>1:break
            if bool(z.at[j,"cross224"]):
                cross_i=j;break
        if cross_i is None:continue
        headroom=float(z.at[cross_i,"ema448"]/z.at[cross_i,"ac"]-1) if z.at[cross_i,"ac"]>0 else np.nan
        gap=float(abs(z.at[cross_i,"ema224"]-z.at[cross_i,"ema112"])/z.at[cross_i,"ema224"])
        if not(0.05<=headroom<=0.35 and gap<=.08):continue
        cross_ctx={
            "event_to_cross_days":int(cross_i-i),
            "cross_headroom448":headroom,
            "cross_gap112_224":gap,
            "cross_ema224_slope20":float(z.at[cross_i,"ema224_slope20"]) if pd.notna(z.at[cross_i,"ema224_slope20"]) else np.nan,
            "cross_ema112_slope20":float(z.at[cross_i,"ema112_slope20"]) if pd.notna(z.at[cross_i,"ema112_slope20"]) else np.nan,
            "cross_anchor_margin":float(z.at[cross_i,"ac"]/anchor-1) if anchor>0 else np.nan,
        }
        if cross_i-cooldown["E2_224_RECOVERY"]>=40:
            add_event(rows,z,"E2_224_RECOVERY",cross_i,i,cost,context=cross_ctx); cooldown["E2_224_RECOVERY"]=cross_i

        # Stage 3: lower-volume pullback that still holds EMA224 and anchor.
        pull_i=None
        event_vol=float(r.volume)
        post_high=-np.inf
        for j in range(cross_i+1,min(cross_i+16,len(z)-1)):
            post_high=max(post_high,float(z.at[j,"ah"]))
            if float(z.at[j,"ac"])<anchor:break
            dist=float(z.at[j,"ac"]/z.at[j,"ema224"]-1)
            dd=float(z.at[j,"ac"]/post_high-1) if post_high>0 else 0
            volcool=float(z.at[j,"volume"])<=event_vol*.60
            if 0<=dist<=.08 and -.20<=dd<=-.03 and volcool:
                pull_i=j;break
        if pull_i is not None:
            pull_ctx=dict(cross_ctx)
            pull_ctx.update({
                "cross_to_pull_days":int(pull_i-cross_i),
                "event_to_pull_days":int(pull_i-i),
                "pull_dist224":float(z.at[pull_i,"ac"]/z.at[pull_i,"ema224"]-1) if z.at[pull_i,"ema224"]>0 else np.nan,
                "pull_drawdown":float(z.at[pull_i,"ac"]/post_high-1) if post_high>0 else np.nan,
                "pull_volume_event_ratio":float(z.at[pull_i,"volume"]/event_vol) if event_vol>0 else np.nan,
                "pull_volume_v20_ratio":float(z.at[pull_i,"vr"]) if pd.notna(z.at[pull_i,"vr"]) else np.nan,
                "pull_anchor_margin":float(z.at[pull_i,"ac"]/anchor-1) if anchor>0 else np.nan,
                "pull_headroom448":float(z.at[pull_i,"ema448"]/z.at[pull_i,"ac"]-1) if z.at[pull_i,"ac"]>0 else np.nan,
                "pull_gap112_224":float(abs(z.at[pull_i,"ema224"]-z.at[pull_i,"ema112"])/z.at[pull_i,"ema224"]) if z.at[pull_i,"ema224"]>0 else np.nan,
                "pull_ema224_slope20":float(z.at[pull_i,"ema224_slope20"]) if pd.notna(z.at[pull_i,"ema224_slope20"]) else np.nan,
                "pull_ema112_slope20":float(z.at[pull_i,"ema112_slope20"]) if pd.notna(z.at[pull_i,"ema112_slope20"]) else np.nan,
                "anchor_price_adj":anchor,
                "target448_signal":float(z.at[pull_i,"ema448"]) if pd.notna(z.at[pull_i,"ema448"]) else np.nan,
            })
            if pull_i-cooldown["E3_CLASSIC_PULLBACK"]>=40:
                add_event(rows,z,"E3_CLASSIC_PULLBACK",pull_i,i,cost,context=pull_ctx)
                cooldown["E3_CLASSIC_PULLBACK"]=pull_i

            # Stage 4: transparent re-acceleration confirmation after the pullback.
            # Generic price/EMA confirmation only; no proprietary Dante indicator is inferred.
            reaccel_i=None
            for j in range(pull_i+1,min(pull_i+11,len(z)-1)):
                if float(z.at[j,"ac"])<anchor:break
                if any(pd.isna(z.at[j,k]) for k in ["ema5","ema15","ema224","prior5_high"]):continue
                above224=bool(z.at[j,"ac"]>=z.at[j,"ema224"])
                reclaim5=bool(z.at[j,"ac"]>z.at[j,"prior5_high"])
                short_bull=bool(z.at[j,"ema5"]>z.at[j,"ema15"])
                if above224 and reclaim5 and short_bull:
                    reaccel_i=j
                    break
            if reaccel_i is not None and reaccel_i-cooldown["E4_REACCELERATION"]>=40:
                reaccel_ctx=dict(pull_ctx)
                reaccel_ctx.update({
                    "pull_to_reaccel_days":int(reaccel_i-pull_i),
                    "event_to_reaccel_days":int(reaccel_i-i),
                    "reaccel_volume_ratio":float(z.at[reaccel_i,"vr"]) if pd.notna(z.at[reaccel_i,"vr"]) else np.nan,
                    "reaccel_dist224":float(z.at[reaccel_i,"ac"]/z.at[reaccel_i,"ema224"]-1) if z.at[reaccel_i,"ema224"]>0 else np.nan,
                    "reaccel_ema5_15_gap":float(z.at[reaccel_i,"ema5"]/z.at[reaccel_i,"ema15"]-1) if z.at[reaccel_i,"ema15"]>0 else np.nan,
                    "reaccel_headroom448":float(z.at[reaccel_i,"ema448"]/z.at[reaccel_i,"ac"]-1) if z.at[reaccel_i,"ac"]>0 else np.nan,
                    "target448_signal":float(z.at[reaccel_i,"ema448"]) if pd.notna(z.at[reaccel_i,"ema448"]) else np.nan,
                })
                add_event(rows,z,"E4_REACCELERATION",reaccel_i,i,cost,context=reaccel_ctx)
                cooldown["E4_REACCELERATION"]=reaccel_i
    return rows

def summarize(ev):
    out=[]
    regimes=[
        ("ALL",ev),
        ("BREADTH20_50",ev[ev.breadth20>=.50]),
        ("BREADTH20_60",ev[ev.breadth20>=.60]),
        ("BOTH50",ev[(ev.breadth20>=.50)&(ev.breadth60>=.50)]),
        ("BOTH60",ev[(ev.breadth20>=.60)&(ev.breadth60>=.60)]),
    ]
    for regime,base in regimes:
        for (sp,k),q in base.groupby(["split","kind"],dropna=False):
            rec={"regime":regime,"split":sp,"kind":k,"n":len(q)}
            for h in (20,60):
                s=q[f"ret{h}"].dropna()
                rec[f"n{h}"]=len(s); rec[f"mean{h}"]=s.mean() if len(s) else np.nan
                rec[f"median{h}"]=s.median() if len(s) else np.nan
                rec[f"win{h}"]=(s>0).mean() if len(s) else np.nan
            out.append(rec)
    return pd.DataFrame(out).sort_values(["regime","split","kind"])

def summarize_yearly(ev):
    out=[]
    if ev.empty:return pd.DataFrame()
    x=ev.copy()
    x["year"]=pd.to_datetime(x["signal_date"],errors="coerce").dt.year
    x["breadth20_bin"]=pd.cut(
        x["breadth20"],
        bins=[-np.inf,.40,.50,.60,np.inf],
        labels=["LT40","40_50","50_60","GE60"],
        right=False,
    )
    groups=[
        ("YEAR_ALL",["year","kind"]),
        ("YEAR_EXCHANGE",["year","exchange","kind"]),
        ("YEAR_QUALITY",["year","event_quality","kind"]),
        ("YEAR_EXCHANGE_QUALITY",["year","exchange","event_quality","kind"]),
        ("YEAR_BREADTH20",["year","breadth20_bin","kind"]),
        ("SPLIT_EXCHANGE",["split","exchange","kind"]),
        ("SPLIT_QUALITY",["split","event_quality","kind"]),
        ("SPLIT_EXCHANGE_QUALITY",["split","exchange","event_quality","kind"]),
    ]
    for view,keys in groups:
        for vals,q in x.groupby(keys,dropna=False):
            if not isinstance(vals,tuple):vals=(vals,)
            rec={"view":view}
            for k,v in zip(keys,vals):rec[k]=v
            rec["n"]=len(q)
            for h in (20,60):
                s=q[f"ret{h}"].dropna()
                rec[f"n{h}"]=len(s)
                rec[f"mean{h}"]=s.mean() if len(s) else np.nan
                rec[f"median{h}"]=s.median() if len(s) else np.nan
                rec[f"win{h}"]=(s>0).mean() if len(s) else np.nan
            out.append(rec)
    return pd.DataFrame(out)


def summarize_robustness(ev):
    """Diagnose regime dependence and winner concentration for E2/E3."""
    out=[]
    if ev.empty:return pd.DataFrame()
    x=ev[ev["kind"].isin(["E2_224_RECOVERY","E3_CLASSIC_PULLBACK","E4_REACCELERATION"])].copy()
    x["year"]=pd.to_datetime(x["signal_date"],errors="coerce").dt.year
    groups=[
        ("SPLIT",["split","kind"]),
        ("YEAR",["year","kind"]),
        ("YEAR_EXCHANGE",["year","exchange","kind"]),
        ("YEAR_QUALITY",["year","event_quality","kind"]),
        ("YEAR_EXCHANGE_QUALITY",["year","exchange","event_quality","kind"]),
    ]
    for view,keys in groups:
        for vals,q in x.groupby(keys,dropna=False,observed=True):
            if not isinstance(vals,tuple):vals=(vals,)
            rec={"view":view}
            for k,v in zip(keys,vals):rec[k]=v
            rec["n"]=len(q)
            for h in (20,60):
                s=q[f"ret{h}"].dropna().sort_values(ascending=False)
                rec[f"n{h}"]=len(s)
                rec[f"mean{h}"]=s.mean() if len(s) else np.nan
                rec[f"median{h}"]=s.median() if len(s) else np.nan
                rec[f"win{h}"]=(s>0).mean() if len(s) else np.nan
                rec[f"max{h}"]=s.iloc[0] if len(s) else np.nan
                rec[f"mean{h}_ex_top1"]=s.iloc[1:].mean() if len(s)>1 else np.nan
                rec[f"mean{h}_ex_top3"]=s.iloc[3:].mean() if len(s)>3 else np.nan
            out.append(rec)
    return pd.DataFrame(out)


def summarize_structure(ev):
    """Describe E3 structure without choosing thresholds from future returns."""
    if ev.empty:return pd.DataFrame()
    x=ev[ev["kind"]=="E3_CLASSIC_PULLBACK"].copy()
    if x.empty:return pd.DataFrame()
    x["ema224_slope_bin"]=pd.cut(
        x["pull_ema224_slope20"],[-np.inf,-.02,0,.02,np.inf],
        labels=["LT_M2","M2_0","0_2","GE2"],right=False,
    )
    x["gap_bin"]=pd.cut(
        x["pull_gap112_224"],[-np.inf,.03,.05,.08,np.inf],
        labels=["LT3","3_5","5_8","GE8"],right=False,
    )
    x["headroom_bin"]=pd.cut(
        x["pull_headroom448"],[-np.inf,.10,.20,.35,np.inf],
        labels=["LT10","10_20","20_35","GE35"],right=False,
    )
    x["event_to_cross_bin"]=pd.cut(
        x["event_to_cross_days"],[-np.inf,5,10,15,np.inf],
        labels=["LE4","5_9","10_14","GE15"],right=False,
    )
    x["cross_to_pull_bin"]=pd.cut(
        x["cross_to_pull_days"],[-np.inf,4,8,12,np.inf],
        labels=["LE3","4_7","8_11","GE12"],right=False,
    )
    x["pullback_bin"]=pd.cut(
        x["pull_drawdown"],[-np.inf,-.12,-.07,-.03,np.inf],
        labels=["LE_M12","M12_M7","M7_M3","GT_M3"],right=False,
    )
    dimensions=[
        ("EVENT_BREAKOUT20","event_breakout20"),
        ("EMA224_SLOPE20","ema224_slope_bin"),
        ("GAP112_224","gap_bin"),
        ("HEADROOM448","headroom_bin"),
        ("EVENT_TO_CROSS","event_to_cross_bin"),
        ("CROSS_TO_PULL","cross_to_pull_bin"),
        ("PULLBACK_DEPTH","pullback_bin"),
    ]
    out=[]
    for view,col in dimensions:
        for (sp,bucket),q in x.groupby(["split",col],dropna=False,observed=True):
            rec={"view":view,"bucket":bucket,"split":sp,"n":len(q)}
            for h in (20,60):
                s=q[f"ret{h}"].dropna()
                rec[f"n{h}"]=len(s)
                rec[f"mean{h}"]=s.mean() if len(s) else np.nan
                rec[f"median{h}"]=s.median() if len(s) else np.nan
                rec[f"win{h}"]=(s>0).mean() if len(s) else np.nan
            out.append(rec)
    return pd.DataFrame(out)


def summarize_regime_dynamics(ev):
    """Test whether E3 depends on an improving market regime, not static breadth alone."""
    if ev.empty:return pd.DataFrame()
    x=ev[ev["kind"]=="E3_CLASSIC_PULLBACK"].copy()
    if x.empty:return pd.DataFrame()
    x["breadth20_chg20_bin"]=pd.cut(
        x["breadth20_chg20"],[-np.inf,-.10,0,.10,np.inf],
        labels=["LE_M10PP","M10_0PP","0_10PP","GE10PP"],right=False,
    )
    x["breadth60_chg20_bin"]=pd.cut(
        x["breadth60_chg20"],[-np.inf,-.10,0,.10,np.inf],
        labels=["LE_M10PP","M10_0PP","0_10PP","GE10PP"],right=False,
    )
    x["market_median20_bin"]=pd.cut(
        x["market_median20"],[-np.inf,-.10,0,.10,np.inf],
        labels=["LT_M10","M10_0","0_10","GE10"],right=False,
    )
    x["market_median60_bin"]=pd.cut(
        x["market_median60"],[-np.inf,-.15,0,.15,np.inf],
        labels=["LT_M15","M15_0","0_15","GE15"],right=False,
    )
    x["breadth_recovery_state"]=np.select(
        [
            (x["breadth20_chg20"]>=.10)&(x["breadth20"]>=.50),
            (x["breadth20_chg20"]>=.10)&(x["breadth20"]<.50),
            (x["breadth20_chg20"]<0)&(x["breadth20"]<.50),
        ],
        ["BROAD_RECOVERY","NARROW_RECOVERY","WEAKENING_WEAK"],
        default="OTHER",
    )
    dimensions=[
        ("BREADTH20_CHG20","breadth20_chg20_bin"),
        ("BREADTH60_CHG20","breadth60_chg20_bin"),
        ("MARKET_MEDIAN20","market_median20_bin"),
        ("MARKET_MEDIAN60","market_median60_bin"),
        ("BREADTH_RECOVERY_STATE","breadth_recovery_state"),
    ]
    out=[]
    for view,col in dimensions:
        for (sp,bucket),q in x.groupby(["split",col],dropna=False,observed=True):
            rec={"view":view,"bucket":bucket,"split":sp,"n":len(q)}
            for h in (20,60):
                s=q[f"ret{h}"].dropna()
                rec[f"n{h}"]=len(s)
                rec[f"mean{h}"]=s.mean() if len(s) else np.nan
                rec[f"median{h}"]=s.median() if len(s) else np.nan
                rec[f"win{h}"]=(s>0).mean() if len(s) else np.nan
            out.append(rec)
    return pd.DataFrame(out)


def summarize_trade_path(ev):
    """Evaluate the actual structural plan: anchor-open invalidation vs signal-day EMA448 target."""
    if ev.empty:return pd.DataFrame()
    x=ev[ev["kind"].isin(["E3_CLASSIC_PULLBACK","E4_REACCELERATION"])].copy()
    if x.empty:return pd.DataFrame()
    x["year"]=pd.to_datetime(x["signal_date"],errors="coerce").dt.year
    out=[]
    for view,keys in [("SPLIT",["split","kind"]),("YEAR",["year","kind"]),("SPLIT_EXCHANGE",["split","exchange","kind"])]:
        for vals,q in x.groupby(keys,dropna=False):
            if not isinstance(vals,tuple):vals=(vals,)
            base={"view":view}
            for k,v in zip(keys,vals):base[k]=v
            for h in (20,60):
                full=q[(q[f"path{h}_valid"]==True)&(q[f"path{h}_full_horizon"]==True)].copy()
                rec=dict(base); rec["horizon"]=h; rec["n"]=len(full)
                if len(full):
                    outcome=full[f"path{h}_outcome"].astype(str)
                    ret=full[f"path{h}_net_return"].astype(float)
                    rm=full[f"path{h}_r_multiple"].astype(float)
                    rec["target_rate"]=(outcome=="TARGET").mean()
                    rec["stop_rate"]=outcome.isin(["STOP","STOP_SAME_DAY"]).mean()
                    rec["timeout_rate"]=(outcome=="TIMEOUT").mean()
                    rec["mean_return"]=ret.mean()
                    rec["median_return"]=ret.median()
                    rec["win_rate"]=(ret>0).mean()
                    rec["mean_r"]=rm.mean()
                    rec["median_r"]=rm.median()
                    rec["mean_mfe"]=full[f"path{h}_mfe"].mean()
                    rec["mean_mae"]=full[f"path{h}_mae"].mean()
                else:
                    for k in ["target_rate","stop_rate","timeout_rate","mean_return","median_return","win_rate","mean_r","median_r","mean_mfe","mean_mae"]:
                        rec[k]=np.nan
                out.append(rec)
    return pd.DataFrame(out)

def main():
    a=parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    panel=pd.concat([load(a.ko,"KO"),load(a.kq,"KQ")],ignore_index=True)
    panel=panel.sort_values(["series_id","date"]).reset_index(drop=True)
    gret=panel.groupby("series_id",sort=False)["adjusted_close"]
    panel["mret20"]=gret.pct_change(20,fill_method=None)
    panel["mret60"]=gret.pct_change(60,fill_method=None)
    b=panel.dropna(subset=["mret20","mret60"]).copy()
    b["up20"]=(b["mret20"]>0).astype(float)
    b["up60"]=(b["mret60"]>0).astype(float)
    breadth=(b.groupby(["exchange","date"],as_index=False)
               .agg(
                   breadth20=("up20","mean"),
                   breadth60=("up60","mean"),
                   market_mean20=("mret20","mean"),
                   market_median20=("mret20","median"),
                   market_mean60=("mret60","mean"),
                   market_median60=("mret60","median"),
               )
               .rename(columns={"date":"signal_date"})
               .sort_values(["exchange","signal_date"]))
    breadth["breadth20_chg20"]=breadth.groupby("exchange",sort=False)["breadth20"].diff(20)
    breadth["breadth60_chg20"]=breadth.groupby("exchange",sort=False)["breadth60"].diff(20)
    start=pd.Timestamp(a.start); cost=a.cost_bps/10000.0
    rows=[]; total=panel.series_id.nunique()
    for n,(sid,g) in enumerate(panel.groupby("series_id",sort=False),1):
        if len(g)>=560:rows.extend(scan_one(g,start,cost))
        if n%250==0 or n==total:print(f"series {n}/{total} events={len(rows):,}",flush=True)
    ev=pd.DataFrame(rows)
    if not ev.empty:
        ev["signal_date"]=pd.to_datetime(ev["signal_date"],errors="coerce")
        ev=ev.merge(breadth,on=["exchange","signal_date"],how="left")
    ev.to_csv(a.out/"event_first_historical_events.csv",index=False,encoding="utf-8-sig")
    sm=summarize(ev) if not ev.empty else pd.DataFrame()
    sm.to_csv(a.out/"event_first_summary.csv",index=False,encoding="utf-8-sig")
    yr=summarize_yearly(ev)
    yr.to_csv(a.out/"event_first_yearly.csv",index=False,encoding="utf-8-sig")
    rb=summarize_robustness(ev)
    rb.to_csv(a.out/"event_first_robustness.csv",index=False,encoding="utf-8-sig")
    st=summarize_structure(ev)
    st.to_csv(a.out/"event_first_structure.csv",index=False,encoding="utf-8-sig")
    rg=summarize_regime_dynamics(ev)
    rg.to_csv(a.out/"event_first_regime_dynamics.csv",index=False,encoding="utf-8-sig")
    tp=summarize_trade_path(ev)
    tp.to_csv(a.out/"event_first_trade_path.csv",index=False,encoding="utf-8-sig")
    print("\n=== EVENT_FIRST SUMMARY (returns net of one round-trip cost assumption) ===")
    if not sm.empty:
        show=sm.copy()
        for c in ["mean20","median20","win20","mean60","median60","win60"]:
            if c in show:show[c]=(show[c]*100).round(2)
        print(show.to_string(index=False))
    if not yr.empty:
        focus=yr[(yr["view"]=="YEAR_ALL") & (yr["kind"].isin(["E2_224_RECOVERY","E3_CLASSIC_PULLBACK","E4_REACCELERATION"]))].copy()
        for c in ["mean20","median20","win20","mean60","median60","win60"]:
            if c in focus:focus[c]=(focus[c]*100).round(2)
        print("\n=== E2/E3/E4 YEARLY DIAGNOSTIC (%) ===")
        print(focus.to_string(index=False))
    if not rb.empty:
        focus=rb[(rb["view"]=="SPLIT") & (rb["kind"]=="E3_CLASSIC_PULLBACK")].copy()
        for c in ["mean20","median20","win20","max20","mean20_ex_top1","mean20_ex_top3",
                  "mean60","median60","win60","max60","mean60_ex_top1","mean60_ex_top3"]:
            if c in focus:focus[c]=(focus[c]*100).round(2)
        print("\n=== E3 ROBUSTNESS / WINNER CONCENTRATION (%) ===")
        print(focus.to_string(index=False))
    if not st.empty:
        show=st.copy()
        for c in ["mean20","median20","win20","mean60","median60","win60"]:
            if c in show:show[c]=(show[c]*100).round(2)
        print("\n=== E3 STRUCTURE DIAGNOSTIC BY SPLIT (%) ===")
        print(show.to_string(index=False))
    if not rg.empty:
        show=rg.copy()
        for c in ["mean20","median20","win20","mean60","median60","win60"]:
            if c in show:show[c]=(show[c]*100).round(2)
        print("\n=== E3 MARKET REGIME DYNAMICS (%) ===")
        print(show.to_string(index=False))
    if not tp.empty:
        show=tp[tp["view"]=="SPLIT"].copy()
        for c in ["target_rate","stop_rate","timeout_rate","mean_return","median_return","win_rate","mean_mfe","mean_mae"]:
            if c in show:show[c]=(show[c]*100).round(2)
        for c in ["mean_r","median_r"]:
            if c in show:show[c]=show[c].round(2)
        print("\n=== E3/E4 STRUCTURAL TRADE PATH: ANCHOR STOP VS EMA448 TARGET ===")
        print(show.to_string(index=False))
if __name__=="__main__":main()