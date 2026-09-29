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
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import FinanceDataReader as fdr

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
    for n in (112,224,448):g[f"ema{n}"]=g.ac.ewm(span=n,adjust=False,min_periods=n).mean()
    g["vol20"]=g.volume.rolling(20,min_periods=20).mean()
    g["amount20"]=g.amount.rolling(20,min_periods=20).mean()
    g["vr"]=g.volume/g.vol20
    g["ret1"]=g.ac.pct_change(fill_method=None)
    g["body"]=(g.ac-g.ao)/g.ao
    rng=(g.ah-g.al).replace(0,np.nan)
    g["close_pos"]=(g.ac-g.al)/rng
    g["upper_wick"]=(g.ah-np.maximum(g.ao,g.ac))/rng
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

    category=None
    if anchor_alive and had_224_cross and currently_above224 and 0<=dist224<=.08 and -0.20<=drawdown<=-.03 and vol_cool:
        category="DANTE_PULLBACK"
    elif anchor_alive and had_224_cross and currently_above224 and 0<=dist224<=.10:
        category="224_HOLD"
    elif anchor_alive and (not currently_above224) and -.06<=dist224<0 and gap112224<=.08:
        category="PRE224_ENERGY"
    elif anchor_alive and currently_above224 and headroom>=.05:
        category="POST_EVENT_TREND"
    if category is None:return None

    score={"DANTE_PULLBACK":40,"224_HOLD":32,"PRE224_ENERGY":25,"POST_EVENT_TREND":20}[category]
    score+=10 if long_below else 0
    score+=10 if gap112224<=.05 else 5 if gap112224<=.08 else 0
    score+=10 if headroom>=.10 else 5 if headroom>=.05 else 0
    score+=8 if vol_cool else 0
    score+=8 if closes_below==0 else 3
    score+=6 if bool(e.ac>e.prior20_high) else 0
    score+=4 if (e.close_pos>=.65 and e.upper_wick<=.35) else 0
    factor_cur=float(cur.ac/cur.close) if float(cur.close)>0 else 1.0
    ema224_raw=float(cur.ema224/factor_cur)
    ema448_raw=float(cur.ema448/factor_cur)
    anchor_open_raw=float(event_open_adj/factor_cur)
    trigger_raw=ema224_raw*1.01
    structural_rr=np.nan
    if trigger_raw>anchor_open_raw and ema448_raw>trigger_raw:
        structural_rr=(ema448_raw-trigger_raw)/(trigger_raw-anchor_open_raw)
    long_below=bool(e.below80>=60) if pd.notna(e.below80) else False
    classic_tier=""
    if anchor_alive and long_below and age>=1:
        if category=="DANTE_PULLBACK" and 3<=headroom*100<=30 and gap112224<=.08:
            classic_tier="A"
        elif category=="224_HOLD" and 5<=headroom*100<=30 and gap112224<=.08:
            classic_tier="B"
        elif category=="PRE224_ENERGY" and 5<=headroom*100<=35 and gap112224<=.08:
            classic_tier="B"
    return {
        "category":category,"classic_tier":classic_tier,"score":score,"age":age,"anchor_alive":anchor_alive,"closes_below_anchor":closes_below,
        "had_224_cross":had_224_cross,"current_above224":currently_above224,"dist224_pct":dist224*100,
        "gap112224_pct":gap112224*100,"headroom448_pct":headroom*100,"drawdown_from_post_high_pct":drawdown*100,
        "volume_cooled":vol_cool,"current_close":float(cur.close),
        "ema112":float(cur.ema112/factor_cur),"ema224":ema224_raw,
        "ema448":ema448_raw,"trigger_224_confirm":trigger_raw,"anchor_invalidation":anchor_open_raw,
        "structural_rr_to_448":structural_rr,
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
    target=pd.Timestamp(a.target_date).normalize() if a.target_date else pd.Timestamp(datetime.now(KST).date()-timedelta(days=1))
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

    rows=[]; watch=[]
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
        rows.append({**refreshed_rec,**cc,"refresh_error":err})
    watch_df=pd.DataFrame(watch)
    if not watch_df.empty:
        watch_df=watch_df.sort_values(["event_date","event_volume_ratio"],ascending=[False,False])
    watch_df.to_csv(a.out/"event_first_watch_rejections.csv",index=False,encoding="utf-8-sig")
    out=pd.DataFrame(rows)
    if not out.empty:
        out=out.sort_values(["score","event_date"],ascending=[False,False]).reset_index(drop=True)
        out["rank"]=np.arange(1,len(out)+1)
    out.to_csv(a.out/"event_first_candidates.csv",index=False,encoding="utf-8-sig")
    print(f"event_first_candidates={len(out):,}",flush=True)
    if not out.empty:
        classic=out[out["classic_tier"].ne("")].copy()
        if not classic.empty:
            print("\n=== CLASSIC_DANTE_CANDIDATES ===")
            print(classic[["rank","code","name","exchange","classic_tier","category","score","event_date","event_ret_pct","event_volume_ratio","current_close","ema112","ema224","ema448","trigger_224_confirm","anchor_invalidation","structural_rr_to_448","dist224_pct","gap112224_pct","headroom448_pct","drawdown_from_post_high_pct","volume_cooled","closes_below_anchor","age"]].head(30).to_string(index=False))
        cols=["rank","code","name","exchange","classic_tier","category","score","event_date","event_ret_pct","event_volume_ratio",
              "current_close","ema112","ema224","ema448","dist224_pct","gap112224_pct","headroom448_pct",
              "drawdown_from_post_high_pct","volume_cooled","closes_below_anchor","age"]
        print(out[cols].head(30).to_string(index=False))
if __name__=="__main__":main()