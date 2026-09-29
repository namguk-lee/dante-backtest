#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Event-first Dante scanner.

Stage 1: find recent abnormal upside/volume events ("energy" / anchor candidates).
Stage 2: only among those stocks, judge whether the post-event structure resembles
public Dante-style patterns: anchor open alive, EMA224 recovery/hold, pullback,
112/224 convergence, and remaining room toward EMA448.

This is a transparent research approximation, not a reproduction of proprietary indicators.
"""
from __future__ import annotations
import argparse, math, random, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import FinanceDataReader as fdr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KST=ZoneInfo("Asia/Seoul")

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--ko",type=Path,required=True); p.add_argument("--kq",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("dante_event_results"))
    p.add_argument("--target-date",default=None)
    p.add_argument("--event-lookback",type=int,default=60)
    p.add_argument("--min-event-return",type=float,default=.05)
    p.add_argument("--min-event-body",type=float,default=.035)
    p.add_argument("--min-event-volume-ratio",type=float,default=2.0)
    p.add_argument("--min-turnover",type=float,default=5_000_000_000)
    p.add_argument("--workers",type=int,default=6)
    p.add_argument("--retries",type=int,default=3)
    return p.parse_args()

def load(path,ex):
    x=pd.read_parquet(path).copy()
    x["date"]=pd.to_datetime(x.date,errors="coerce")
    x["code"]=x.code.astype(str).str.replace(r"\.0$","",regex=True).str.zfill(6)
    if "exchange" not in x:x["exchange"]=ex
    if "name" not in x:x["name"]=""
    if "adjusted_close" not in x:x["adjusted_close"]=x["close"]
    if "amount" not in x:x["amount"]=x["close"]*x["volume"]
    return x.sort_values(["series_id","date"])

def feat(g):
    g=g.sort_values("date").copy()
    f=(g.adjusted_close/g.close).replace([np.inf,-np.inf],np.nan).ffill().bfill().fillna(1)
    g["ao"]=g.open*f; g["ah"]=g.high*f; g["al"]=g.low*f; g["ac"]=g.adjusted_close
    for n in (5,15,112,224,448):g[f"ema{n}"]=g.ac.ewm(span=n,adjust=False,min_periods=n).mean()
    g["vol20"]=g.volume.rolling(20,min_periods=20).mean()
    g["amount20"]=g.amount.rolling(20,min_periods=20).mean()
    g["vr"]=g.volume/g.vol20
    g["ret1"]=g.ac.pct_change(fill_method=None)
    g["body"]=(g.ac-g.ao)/g.ao
    rng=(g.ah-g.al).replace(0,np.nan)
    g["close_pos"]=(g.ac-g.al)/rng
    g["upper_wick"]=(g.ah-np.maximum(g.ao,g.ac))/rng
    g["prior1_high"]=g.ah.shift(1)
    g["prior3_high"]=g.ah.shift(1).rolling(3,min_periods=3).max()
    g["prior5_high"]=g.ah.shift(1).rolling(5,min_periods=5).max()
    g["prior20_high"]=g.ah.shift(1).rolling(20,min_periods=20).max()
    g["cross224"]=(g.ac>g.ema224)&(g.ac.shift(1)<=g.ema224.shift(1))
    g["below80"]=(g.ac<g.ema224).shift(1).rolling(80,min_periods=80).sum()
    return g

def event_mask(z,a):
    return (
        (z.ret1>=a.min_event_return) &
        (z.body>=a.min_event_body) &
        (z.vr>=a.min_event_volume_ratio) &
        (z.amount20>=a.min_turnover)
    )

def find_stage1(panel,a):
    base_date=panel.date.max(); rows=[]
    active_ids=set(panel.loc[panel.date.eq(base_date),"series_id"].astype(str))
    for sid,g in panel[panel.series_id.astype(str).isin(active_ids)].groupby("series_id",sort=False):
        z=feat(g)
        if len(z)<448:continue
        tail=z.tail(a.event_lookback)
        ev=tail[event_mask(tail,a)]
        if ev.empty:continue
        # Favor the most recent meaningful event. Keep strongest stats for diagnostics.
        e=ev.iloc[-1]; recent_high=float(z.loc[e.name:,"ah"].max()) if e.name in z.index else float(e.ah)
        quality="CLEAN_BREAKOUT" if (e.close_pos>=.65 and e.upper_wick<=.35) else "WICK_ENERGY"
        rows.append({
            "series_id":sid,"code":str(e.code).zfill(6),"exchange":e.exchange,"name":e.get("name",""),
            "base_date":base_date,"event_date":e.date,"event_open":float(e.open),"event_close":float(e.close),
            "event_ret_pct":float(e.ret1*100),"event_body_pct":float(e.body*100),
            "event_volume_ratio":float(e.vr),"event_quality":quality,"event_close_pos":float(e.close_pos),
            "event_upper_wick":float(e.upper_wick),"event_amount20":float(e.amount20),
            "event_above_prior20_high":bool(e.ac>e.prior20_high if pd.notna(e.prior20_high) else False),
            "event_cross224":bool(e.cross224),"event_ema224":float(e.ema224) if pd.notna(e.ema224) else np.nan,
            "recent_high_from_event":recent_high
        })
    return pd.DataFrame(rows)

def fetch_recent(rec,start,end,retries):
    last=None
    for k in range(retries):
        try:
            q=fdr.DataReader("NAVER:"+rec["code"],start.strftime("%Y-%m-%d"),(end+pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
            if q is None or q.empty:return rec["series_id"],pd.DataFrame(),"empty"
            q=q.reset_index(); q.columns=[str(c).lower() for c in q.columns]
            ren={"date":"date","open":"open","high":"high","low":"low","close":"close","volume":"volume","change":"change"}
            q=q.rename(columns=ren)
            need=["date","open","high","low","close","volume"]
            if any(c not in q.columns for c in need):raise ValueError(f"columns={list(q.columns)}")
            q["date"]=pd.to_datetime(q.date,errors="coerce")
            for c in need[1:]+(["change"] if "change" in q.columns else []):q[c]=pd.to_numeric(q[c],errors="coerce")
            q=q.dropna(subset=need); q=q[(q.date>=start)&(q.date<=end)]
            q["amount"]=q.close*q.volume
            return rec["series_id"],q.sort_values("date"),None
        except Exception as exc:
            last=repr(exc); time.sleep((k+1)*.8+random.random()*.3)
    return rec["series_id"],pd.DataFrame(),last

def stitch_one(hist,q):
    if q.empty:return hist
    q=q.sort_values("date").drop_duplicates("date",keep="last").copy()
    qstart=pd.Timestamp(q.date.min())
    older=hist[hist.date<qstart].sort_values("date").copy()
    if older.empty:
        return hist
    prev=older.iloc[-1]; raw=float(prev.close); adj=float(prev.adjusted_close); rows=[]
    for rec in q.to_dict("records"):
        rr=float(rec["close"]/raw-1) if raw>0 else 0.0
        ch=rec.get("change",np.nan)
        eff=float(ch) if pd.notna(ch) and abs(float(ch))<1 and abs(rr-float(ch))>.03 else rr
        eff=float(np.clip(eff,-.95,5))
        adj=adj*(1+eff)
        rows.append({
            "series_id":prev.series_id,"code":prev.code,"exchange":prev.exchange,"name":prev.get("name",""),
            "date":rec["date"],"open":rec["open"],"high":rec["high"],"low":rec["low"],"close":rec["close"],
            "adjusted_close":adj,"volume":rec["volume"],"amount":rec["amount"]
        })
        raw=float(rec["close"])
    out=pd.concat([older,pd.DataFrame(rows)],ignore_index=True,sort=False)
    return out.sort_values("date").drop_duplicates("date",keep="last")


def render_candidate_chart(z,rec,out_dir,bars=120):
    """Render an explainable chart for CLASSIC watch candidates."""
    t=z.sort_values("date").tail(bars).copy().reset_index(drop=True)
    if t.empty:return
    cur=t.iloc[-1]
    factor=float(cur.ac/cur.close) if float(cur.close)>0 else 1.0
    for src,dst in [("ao","o"),("ah","h"),("al","l"),("ac","c"),
                    ("ema112","e112"),("ema224","e224"),("ema448","e448")]:
        t[dst]=t[src]/factor
    x=np.arange(len(t))
    fig,(ax,av)=plt.subplots(2,1,figsize=(14,8),sharex=True,gridspec_kw={"height_ratios":[4,1]})
    for i,r in t.iterrows():
        ax.vlines(i,r.l,r.h,linewidth=.8)
        ax.hlines(r.o,i-.32,i,linewidth=.8)
        ax.hlines(r.c,i,i+.32,linewidth=.8)
    ax.plot(x,t.e112,label="EMA112",linewidth=1.1)
    ax.plot(x,t.e224,label="EMA224",linewidth=1.2)
    ax.plot(x,t.e448,label="EMA448",linewidth=1.2)

    event_date=pd.Timestamp(rec.get("event_date")) if pd.notna(rec.get("event_date")) else None
    pull_date=pd.Timestamp(rec.get("pullback_date")) if pd.notna(rec.get("pullback_date")) else None
    if event_date is not None:
        hit=t.index[t.date.eq(event_date)]
        if len(hit):ax.axvline(int(hit[-1]),linestyle="--",linewidth=1,label="Event")
    if pull_date is not None:
        hit=t.index[t.date.eq(pull_date)]
        if len(hit):ax.axvline(int(hit[-1]),linestyle=":",linewidth=1.2,label="Pullback")

    levels=[
        ("Anchor invalid",rec.get("anchor_invalidation")),
        ("224 confirm",rec.get("trigger_224_confirm")),
    ]
    if bool(rec.get("pullback_seen",False)):
        levels.extend([
            ("E4 1D",rec.get("reaccel_trigger_1d")),
            ("E4 3D",rec.get("reaccel_trigger_3d")),
            ("E4 5D",rec.get("reaccel_trigger_5d")),
        ])
    for label,val in levels:
        if pd.notna(val):
            ax.axhline(float(val),linewidth=.7,alpha=.55)
            ax.text(len(t)-1,float(val),f" {label} {float(val):,.0f}",fontsize=8,va="bottom")

    av.bar(x,t.volume)
    av.set_ylabel("Volume")
    ax.set_ylabel("Price")
    ax.grid(alpha=.15)
    av.grid(alpha=.15)
    ax.legend(loc="upper left",ncol=5,fontsize=8)

    step=max(1,len(t)//8)
    ticks=list(range(0,len(t),step))
    if ticks[-1]!=len(t)-1:ticks.append(len(t)-1)
    av.set_xticks(ticks)
    av.set_xticklabels([pd.Timestamp(t.at[i,"date"]).strftime("%m-%d") for i in ticks],rotation=0)

    title=(f'{rec.get("code","")} | '
           f'{rec.get("classic_tier","")} {rec.get("action_status","")} | {rec.get("category","")}')
    ax.set_title(title)
    note=(f'event {pd.Timestamp(rec.get("event_date")).date()} '
          f'+{float(rec.get("event_ret_pct",np.nan)):.1f}% vol x{float(rec.get("event_volume_ratio",np.nan)):.1f} | '
          f'224 dist {float(rec.get("dist224_pct",np.nan)):.1f}% | '
          f'448 room {float(rec.get("headroom448_pct",np.nan)):.1f}% | '
          f'anchor closes below {int(rec.get("closes_below_anchor",0))}')
    fig.text(.01,.01,note,fontsize=8)
    fig.tight_layout(rect=[0,.03,1,1])
    out_dir.mkdir(parents=True,exist_ok=True)
    fig.savefig(out_dir/f'{rec.get("code","unknown")}_{rec.get("category","candidate")}.png',dpi=150)
    plt.close(fig)

def classify(z,event_date):
    z=z.sort_values("date").copy()
    cur=z.iloc[-1]
    evs=z[z.date.eq(pd.Timestamp(event_date))]
    if evs.empty:return None
    e=evs.iloc[-1]
    post=z[z.date>=e.date]
    age=len(post)-1
    event_open_adj=float(e.ao)
    closes_below=int((post.ac<event_open_adj).sum())
    anchor_alive=bool(cur.ac>=event_open_adj and closes_below<=1)
    had_224_cross=bool(post.cross224.any())
    currently_above224=bool(cur.ac>=cur.ema224)
    dist224=float(cur.ac/cur.ema224-1) if pd.notna(cur.ema224) else np.nan
    gap112224=float(abs(cur.ema224-cur.ema112)/cur.ema224) if pd.notna(cur.ema112) and pd.notna(cur.ema224) else np.nan
    headroom=float(cur.ema448/cur.ac-1) if pd.notna(cur.ema448) else np.nan
    post_high=float(post.ah.max()); drawdown=float(cur.ac/post_high-1)
    vol_cool=bool(cur.volume<=e.volume*0.60) if e.volume>0 else False
    long_below=bool(e.below80>=60) if pd.notna(e.below80) else False

    # Reconstruct the transparent E3 pullback, then ask whether the latest bar
    # is a post-pullback re-acceleration confirmation (E4 research stage).
    pullback_seen=False
    pullback_date=pd.NaT
    days_since_pull=np.nan
    reaccel_early=False
    reaccel_confirmed=False
    reaccel_strong=False
    if had_224_cross:
        p=post.reset_index(drop=True)
        crosses=np.flatnonzero(p.cross224.fillna(False).to_numpy())
        if len(crosses):
            c0=int(crosses[0]); running_high=-np.inf; pull_local=None
            for jj in range(c0+1,len(p)):
                running_high=max(running_high,float(p.at[jj,"ah"]))
                if float(p.at[jj,"ac"])<event_open_adj:break
                if pd.isna(p.at[jj,"ema224"]) or p.at[jj,"ema224"]<=0:continue
                d224=float(p.at[jj,"ac"]/p.at[jj,"ema224"]-1)
                ddj=float(p.at[jj,"ac"]/running_high-1) if running_high>0 else 0
                vcool=bool(float(p.at[jj,"volume"])<=float(e.volume)*.60) if e.volume>0 else False
                if 0<=d224<=.08 and -.20<=ddj<=-.03 and vcool:
                    pull_local=jj
                    break
            if pull_local is not None:
                pullback_seen=True
                pullback_date=p.at[pull_local,"date"]
                days_since_pull=int(len(p)-1-pull_local)
                common_reaccel=bool(
                    1<=days_since_pull<=10 and anchor_alive and currently_above224 and
                    pd.notna(cur.ema5) and pd.notna(cur.ema15) and cur.ema5>cur.ema15
                )
                reaccel_early=bool(common_reaccel and pd.notna(cur.prior1_high) and cur.ac>cur.prior1_high)
                reaccel_confirmed=bool(common_reaccel and pd.notna(cur.prior3_high) and cur.ac>cur.prior3_high)
                reaccel_strong=bool(common_reaccel and pd.notna(cur.prior5_high) and cur.ac>cur.prior5_high)

    category=None
    if reaccel_strong:
        category="REACCEL_STRONG"
    elif reaccel_confirmed:
        category="REACCEL_CONFIRMED"
    elif reaccel_early:
        category="REACCEL_EARLY"
    elif anchor_alive and had_224_cross and currently_above224 and 0<=dist224<=.08 and -0.20<=drawdown<=-.03 and vol_cool:
        category="DANTE_PULLBACK"
    elif anchor_alive and had_224_cross and currently_above224 and 0<=dist224<=.10:
        category="224_HOLD"
    elif anchor_alive and (not currently_above224) and -.06<=dist224<0 and gap112224<=.08:
        category="PRE224_ENERGY"
    elif anchor_alive and currently_above224 and headroom>=.05:
        category="POST_EVENT_TREND"
    if category is None:return None

    score={"REACCEL_STRONG":48,"REACCEL_CONFIRMED":46,"REACCEL_EARLY":43,"DANTE_PULLBACK":40,"224_HOLD":32,"PRE224_ENERGY":25,"POST_EVENT_TREND":20}[category]
    score+=10 if long_below else 0
    score+=10 if gap112224<=.05 else 5 if gap112224<=.08 else 0
    score+=10 if headroom>=.10 else 5 if headroom>=.05 else 0
    score+=8 if vol_cool else 0
    score+=8 if closes_below==0 else 3
    score+=6 if bool(e.ac>e.prior20_high) else 0
    score+=4 if (e.close_pos>=.65 and e.upper_wick<=.35) else 0
    factor_cur=float(cur.ac/cur.close) if float(cur.close)>0 else 1.0
    ema5_raw=float(cur.ema5/factor_cur) if pd.notna(cur.ema5) else np.nan
    ema15_raw=float(cur.ema15/factor_cur) if pd.notna(cur.ema15) else np.nan
    ema224_raw=float(cur.ema224/factor_cur)
    ema448_raw=float(cur.ema448/factor_cur)
    prior1_raw=float(cur.prior1_high/factor_cur) if pd.notna(cur.prior1_high) else np.nan
    prior3_raw=float(cur.prior3_high/factor_cur) if pd.notna(cur.prior3_high) else np.nan
    prior5_raw=float(cur.prior5_high/factor_cur) if pd.notna(cur.prior5_high) else np.nan
    anchor_open_raw=float(event_open_adj/factor_cur)
    trigger_raw=ema224_raw*1.01
    reaccel_trigger_1d=max(prior1_raw,ema224_raw) if pd.notna(prior1_raw) else ema224_raw
    reaccel_trigger_3d=max(prior3_raw,ema224_raw) if pd.notna(prior3_raw) else ema224_raw
    reaccel_trigger_5d=max(prior5_raw,ema224_raw) if pd.notna(prior5_raw) else ema224_raw
    reaccel_trigger_gap=float(reaccel_trigger_3d/float(cur.close)-1) if float(cur.close)>0 else np.nan
    ema5_above15_now=bool(pd.notna(cur.ema5) and pd.notna(cur.ema15) and cur.ema5>cur.ema15)
    reaccel_window_days_left=max(0,10-int(days_since_pull)) if pullback_seen and pd.notna(days_since_pull) else np.nan
    structural_rr=np.nan
    if trigger_raw>anchor_open_raw and ema448_raw>trigger_raw:
        structural_rr=(ema448_raw-trigger_raw)/(trigger_raw-anchor_open_raw)
    current_rr_to_448=np.nan
    if float(cur.close)>anchor_open_raw and ema448_raw>float(cur.close):
        current_rr_to_448=(ema448_raw-float(cur.close))/(float(cur.close)-anchor_open_raw)
    reaccel3_rr_to_448=np.nan
    if reaccel_trigger_3d>anchor_open_raw and ema448_raw>reaccel_trigger_3d:
        reaccel3_rr_to_448=(ema448_raw-reaccel_trigger_3d)/(reaccel_trigger_3d-anchor_open_raw)
    reaccel_1d_before_448=bool(reaccel_trigger_1d<ema448_raw)
    reaccel_3d_before_448=bool(reaccel_trigger_3d<ema448_raw)
    reaccel_5d_before_448=bool(reaccel_trigger_5d<ema448_raw)
    long_below=bool(e.below80>=60) if pd.notna(e.below80) else False
    classic_tier=""
    if anchor_alive and long_below and age>=1:
        if category in ("REACCEL_STRONG","REACCEL_CONFIRMED","REACCEL_EARLY") and 3<=headroom*100<=30 and gap112224<=.08:
            classic_tier="A"
        elif category=="DANTE_PULLBACK" and 3<=headroom*100<=30 and gap112224<=.08:
            classic_tier="A"
        elif category=="224_HOLD" and 5<=headroom*100<=30 and gap112224<=.08:
            classic_tier="B"
        elif category=="PRE224_ENERGY" and 5<=headroom*100<=35 and gap112224<=.08:
            classic_tier="B"
    if not classic_tier:
        action_status="RESEARCH_ONLY"
    elif category=="REACCEL_STRONG":
        action_status="STRONG_CONFIRM_WATCH"
    elif category=="REACCEL_CONFIRMED":
        action_status="CONFIRMED_WATCH"
    elif category=="REACCEL_EARLY":
        action_status="EARLY_REACCEL_WATCH"
    elif category=="DANTE_PULLBACK":
        if headroom<.05 or not reaccel_1d_before_448:
            action_status="WATCH_TARGET_TOO_CLOSE"
        else:
            action_status="WATCH_REACCEL_CONFIRM"
    elif category=="224_HOLD":
        action_status="WATCH_PULLBACK_CONFIRM"
    elif category=="PRE224_ENERGY":
        action_status="WATCH_224_CONFIRM"
    else:
        action_status="WATCH_STRUCTURE"
    return {
        "category":category,"classic_tier":classic_tier,"action_status":action_status,"score":score,"age":age,"anchor_alive":anchor_alive,"closes_below_anchor":closes_below,
        "had_224_cross":had_224_cross,"pullback_seen":pullback_seen,"pullback_date":pullback_date,
        "days_since_pull":days_since_pull,"reaccel_early":reaccel_early,
        "reaccel_confirmed":reaccel_confirmed,"reaccel_strong":reaccel_strong,
        "current_above224":currently_above224,"dist224_pct":dist224*100,
        "gap112224_pct":gap112224*100,"headroom448_pct":headroom*100,"drawdown_from_post_high_pct":drawdown*100,
        "volume_cooled":vol_cool,"current_close":float(cur.close),
        "ema5":ema5_raw,"ema15":ema15_raw,"ema5_above15_now":ema5_above15_now,
        "ema112":float(cur.ema112/factor_cur),"ema224":ema224_raw,
        "ema448":ema448_raw,"trigger_224_confirm":trigger_raw,
        "prior1_high":prior1_raw,"prior3_high":prior3_raw,"prior5_high":prior5_raw,
        "reaccel_trigger_1d":reaccel_trigger_1d,"reaccel_trigger_3d":reaccel_trigger_3d,
        "reaccel_trigger_5d":reaccel_trigger_5d,"reaccel_trigger_close":reaccel_trigger_3d,
        "reaccel_trigger_gap_pct":reaccel_trigger_gap*100 if pd.notna(reaccel_trigger_gap) else np.nan,
        "reaccel_1d_before_448":reaccel_1d_before_448,"reaccel_3d_before_448":reaccel_3d_before_448,
        "reaccel_5d_before_448":reaccel_5d_before_448,
        "reaccel_window_days_left":reaccel_window_days_left,
        "anchor_invalidation":anchor_open_raw,
        "structural_rr_to_448":structural_rr,
        "current_rr_to_448":current_rr_to_448,"reaccel3_rr_to_448":reaccel3_rr_to_448,
        "current_volume_ratio":float(cur.vr) if pd.notna(cur.vr) else np.nan,
        "current_turnover20":float(cur.amount20) if pd.notna(cur.amount20) else np.nan
    }

def diagnose_rejection(z,event_date):
    z=z.sort_values("date").copy()
    cur=z.iloc[-1]
    evs=z[z.date.eq(pd.Timestamp(event_date))]
    if evs.empty:
        return {"reject_reason":"event_missing_after_refresh"}
    e=evs.iloc[-1]; post=z[z.date>=e.date]
    anchor=float(e.ao)
    closes_below=int((post.ac<anchor).sum())
    anchor_alive=bool(cur.ac>=anchor and closes_below<=1)
    had_cross=bool(post.cross224.any())
    above224=bool(cur.ac>=cur.ema224) if pd.notna(cur.ema224) else False
    dist=float(cur.ac/cur.ema224-1) if pd.notna(cur.ema224) else np.nan
    gap=float(abs(cur.ema224-cur.ema112)/cur.ema224) if pd.notna(cur.ema112) and pd.notna(cur.ema224) else np.nan
    head=float(cur.ema448/cur.ac-1) if pd.notna(cur.ema448) else np.nan
    factor=float(cur.ac/cur.close) if float(cur.close)>0 else 1.0
    reasons=[]
    if not anchor_alive:reasons.append("anchor_broken")
    if not had_cross:reasons.append("no_224_cross_yet")
    if above224 and head<.05:reasons.append("ema448_no_room")
    if (not above224) and not(-.06<=dist<0):reasons.append("too_far_from_224")
    if pd.notna(gap) and gap>.08:reasons.append("112_224_not_converged")
    if not reasons:reasons.append("post_event_structure_incomplete")
    return {
        "reject_reason":"|".join(reasons),
        "current_close":float(cur.close),
        "ema112":float(cur.ema112/factor) if pd.notna(cur.ema112) else np.nan,
        "ema224":float(cur.ema224/factor) if pd.notna(cur.ema224) else np.nan,
        "ema448":float(cur.ema448/factor) if pd.notna(cur.ema448) else np.nan,
        "dist224_pct":dist*100 if pd.notna(dist) else np.nan,
        "gap112224_pct":gap*100 if pd.notna(gap) else np.nan,
        "headroom448_pct":head*100 if pd.notna(head) else np.nan,
        "anchor_alive":anchor_alive,
        "had_224_cross":had_cross,
        "closes_below_anchor":closes_below,
    }

def main():
    a=parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    ko=load(a.ko,"KO"); kq=load(a.kq,"KQ"); panel=pd.concat([ko,kq],ignore_index=True)
    base_date=pd.Timestamp(panel.date.max()).normalize()
    if a.target_date:
        target=pd.Timestamp(a.target_date).normalize()
    else:
        now=datetime.now(KST)
        closed_day=now.date() if now.time()>=dtime(16,10) else now.date()-timedelta(days=1)
        target=pd.Timestamp(closed_day).normalize()
    stage1=find_stage1(panel,a)
    stage1.to_csv(a.out/"stage1_recent_strong_events.csv",index=False,encoding="utf-8-sig")
    print(f"base={base_date.date()} stage1_events={len(stage1):,} target<={target.date()}",flush=True)
    if stage1.empty:
        pd.DataFrame().to_csv(a.out/"event_first_candidates.csv",index=False); return

    refresh_start=base_date-pd.Timedelta(days=120)
    results=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futs=[pool.submit(fetch_recent,r,refresh_start,target,a.retries) for r in stage1.to_dict("records")]
        for i,f in enumerate(as_completed(futs),1):
            results.append(f.result())
            if i%50==0 or i==len(futs):print(f"refresh {i}/{len(futs)}",flush=True)
    res={sid:(q,err) for sid,q,err in results}

    rows=[]; watch=[]; chart_inputs=[]
    for rec in stage1.to_dict("records"):
        hist=panel[panel.series_id.eq(rec["series_id"])].sort_values("date").copy()
        q,err=res.get(rec["series_id"],(pd.DataFrame(),"missing"))
        z=feat(stitch_one(hist,q))
        tail=z.tail(a.event_lookback)
        ev=tail[event_mask(tail,a)]
        if ev.empty:
            watch.append({**rec,"reject_reason":"no_strong_event_after_naver_refresh","refresh_error":err})
            continue
        e=ev.iloc[-1]
        quality="CLEAN_BREAKOUT" if (e.close_pos>=.65 and e.upper_wick<=.35) else "WICK_ENERGY"
        refreshed_rec={
            **rec,
            "event_date":e.date,
            "event_open":float(e.open),
            "event_close":float(e.close),
            "event_ret_pct":float(e.ret1*100),
            "event_body_pct":float(e.body*100),
            "event_volume_ratio":float(e.vr),
            "event_quality":quality,
            "event_close_pos":float(e.close_pos),
            "event_upper_wick":float(e.upper_wick),
            "event_amount20":float(e.amount20),
            "event_above_prior20_high":bool(e.ac>e.prior20_high if pd.notna(e.prior20_high) else False),
            "event_cross224":bool(e.cross224),
            "event_ema224":float(e.ema224) if pd.notna(e.ema224) else np.nan,
            "recent_high_from_event":float(z.loc[e.name:,"ah"].max()) if e.name in z.index else float(e.ah),
        }
        cc=classify(z,e.date)
        if not cc:
            watch.append({**refreshed_rec,**diagnose_rejection(z,e.date),"refresh_error":err})
            continue
        combined={**refreshed_rec,**cc,"refresh_error":err}
        rows.append(combined)
        if combined.get("classic_tier",""):
            chart_inputs.append((combined,z.copy()))
    watch_df=pd.DataFrame(watch)
    if not watch_df.empty:
        watch_df=watch_df.sort_values(["event_date","event_volume_ratio"],ascending=[False,False])
    watch_df.to_csv(a.out/"event_first_watch_rejections.csv",index=False,encoding="utf-8-sig")
    out=pd.DataFrame(rows)
    if not out.empty:
        out=out.sort_values(["score","event_date"],ascending=[False,False]).reset_index(drop=True)
        out["rank"]=np.arange(1,len(out)+1)
    out.to_csv(a.out/"event_first_candidates.csv",index=False,encoding="utf-8-sig")
    charts_dir=a.out/"charts"
    for rec,z in chart_inputs:
        try:
            render_candidate_chart(z,rec,charts_dir)
        except Exception as exc:
            print(f'chart_failed code={rec.get("code")} error={exc!r}',flush=True)
    print(f"event_first_candidates={len(out):,} charts={len(chart_inputs):,}",flush=True)
    if not out.empty:
        classic=out[out["classic_tier"].ne("")].copy()
        if not classic.empty:
            print("\n=== CLASSIC_DANTE_CANDIDATES ===")
            print(classic[["rank","code","name","exchange","classic_tier","action_status","category","score","event_date","event_ret_pct","event_volume_ratio","pullback_date","days_since_pull","reaccel_window_days_left","reaccel_early","reaccel_confirmed","reaccel_strong","current_close","prior1_high","prior3_high","prior5_high","reaccel_trigger_1d","reaccel_trigger_3d","reaccel_trigger_5d","reaccel_trigger_gap_pct","ema5","ema15","ema5_above15_now","ema112","ema224","ema448","trigger_224_confirm","anchor_invalidation","structural_rr_to_448","current_rr_to_448","reaccel3_rr_to_448","dist224_pct","gap112224_pct","headroom448_pct","drawdown_from_post_high_pct","volume_cooled","closes_below_anchor","age"]].head(30).to_string(index=False))
        cols=["rank","code","name","exchange","classic_tier","action_status","category","score","event_date","event_ret_pct","event_volume_ratio",
              "pullback_date","days_since_pull","reaccel_window_days_left","reaccel_early","reaccel_confirmed","reaccel_strong","current_close",
              "prior1_high","prior3_high","prior5_high","reaccel_trigger_1d","reaccel_trigger_3d","reaccel_trigger_5d","reaccel_trigger_gap_pct","ema5","ema15","ema5_above15_now",
              "ema112","ema224","ema448","dist224_pct","gap112224_pct","headroom448_pct",
              "drawdown_from_post_high_pct","volume_cooled","closes_below_anchor","age"]
        print(out[cols].head(30).to_string(index=False))
if __name__=="__main__":main()