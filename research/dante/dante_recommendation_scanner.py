#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Point-in-time Dante-style recommendation scanner.

Produces candidates only; it does not place orders.
Categories:
  STRONG_BREAKOUT - EMA224 cross with volume + candle/resistance confirmation
  BREAKOUT        - base EMA224/volume breakout
  PULLBACK        - recent breakout holding EMA224 on lower volume
  NEAR_BREAKOUT   - compressed reverse-alignment setup approaching EMA224
"""
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--ko",type=Path,required=True); p.add_argument("--kq",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("dante_results"))
    p.add_argument("--asof",default=None)
    p.add_argument("--top",type=int,default=30)
    p.add_argument("--min-turnover",type=float,default=5_000_000_000)
    return p.parse_args()

def load(path,ex):
    x=pd.read_parquet(path).copy()
    if "exchange" not in x:x["exchange"]=ex
    if "adjusted_close" not in x:x["adjusted_close"]=x["close"]
    if "name" not in x:x["name"]=""
    if "amount" not in x:x["amount"]=x["close"]*x["volume"]
    x["date"]=pd.to_datetime(x["date"],errors="coerce")
    for c in ["open","high","low","close","adjusted_close","volume","amount"]:
        x[c]=pd.to_numeric(x[c],errors="coerce")
    x=x.dropna(subset=["series_id","date","open","high","low","close","adjusted_close","volume"])
    return x.sort_values(["exchange","series_id","date"])

def features(g):
    g=g.sort_values("date").copy()
    f=(g.adjusted_close/g.close).replace([np.inf,-np.inf],np.nan).ffill().bfill().fillna(1)
    g["ao"]=g.open*f; g["ah"]=g.high*f; g["al"]=g.low*f; g["ac"]=g.adjusted_close
    for n in [112,224,448]:g[f"ema{n}"]=g.ac.ewm(span=n,adjust=False,min_periods=n).mean()
    g["vol20"]=g.volume.rolling(20,min_periods=20).mean()
    amt=g.amount.where(g.amount>0,g.close*g.volume); g["turnover20"]=amt.rolling(20,min_periods=20).mean()
    g["ret20"]=g.ac.pct_change(20,fill_method=None); g["ret60"]=g.ac.pct_change(60,fill_method=None)
    g["prior20_close_high"]=g.ac.shift(1).rolling(20,min_periods=20).max()
    g["prior20_high"]=g.ah.shift(1).rolling(20,min_periods=20).max()
    rng=(g.ah-g.al).replace(0,np.nan)
    g["body_pct"]=(g.ac-g.ao)/g.ao
    g["close_pos"]=(g.ac-g.al)/rng
    g["upper_wick_ratio"]=(g.ah-np.maximum(g.ao,g.ac))/rng
    g["volume_ratio"]=g.volume/g.vol20
    g["conv_gap"]=(g.ema224-g.ema112)/g.ema224
    g["headroom448"]=g.ema448/g.ac-1
    g["dist224"]=g.ac/g.ema224-1
    g["cross224"]=(g.ac>g.ema224)&(g.ac.shift(1)<=g.ema224.shift(1))
    g["below80"]=(g.ac<g.ema224).shift(1).rolling(80,min_periods=80).sum()
    return g

def clip01(x):return max(0.0,min(1.0,float(x)))

def main():
    a=args(); a.out.mkdir(parents=True,exist_ok=True)
    panel=pd.concat([load(a.ko,"KO"),load(a.kq,"KQ")],ignore_index=True)
    if a.asof:
        asof=pd.Timestamp(a.asof); panel=panel[panel.date<=asof]
    last_date=panel.date.max()
    # market breadth is point-in-time by exchange; computed before candidate filtering
    enriched=[]; rows=[]
    for (_,sid),g in panel.groupby(["exchange","series_id"],sort=False):
        z=features(g)
        if len(z)<448: continue
        enriched.append(z[["exchange","series_id","date","ret20","ret60"]].tail(1))
        i=len(z)-1; r=z.iloc[i]
        if pd.Timestamp(r.date)!=last_date: continue
        if not all(math.isfinite(float(r[c])) for c in ["ema112","ema224","ema448","vol20","turnover20"]):continue
        rev=bool(r.ema112<r.ema224<r.ema448)
        conv=bool(0<=r.conv_gap<=.05)
        room=bool(r.headroom448>=.05)
        liquid=bool(r.turnover20>=a.min_turnover)
        bullish=bool(r.ac>r.ao)
        vbase=bool(1.5<=r.volume_ratio<=5)
        base=bool(r.cross224 and r.below80>=60 and rev and bullish and vbase and liquid)
        strong=bool(base and r.volume_ratio>=2 and r.body_pct>=.02 and r.ac>r.prior20_close_high and r.close_pos>=.70 and r.upper_wick_ratio<=.30 and room)
        near=bool((-0.03<=r.dist224<0) and r.below80>=60 and rev and conv and room and liquid)
        # recent breakout / pullback using only history available as-of
        pull=False; breakout_age=np.nan
        if i>=2:
            lo=max(0,i-10)
            hist=z.iloc[lo:i]
            hits=hist[(hist.cross224)&(hist.below80>=60)&(hist.ema112<hist.ema224)&(hist.ema224<hist.ema448)&(hist.volume_ratio>=1.5)&(hist.volume_ratio<=5)&(hist.ac>hist.ao)]
            if not hits.empty:
                h=hits.iloc[-1]; breakout_age=i-h.name if isinstance(h.name,(int,np.integer)) else len(hist)-1
                pull=bool(0<=r.dist224<=.05 and r.ac>=r.ema224 and r.volume< h.volume and bullish and liquid)
        category="STRONG_BREAKOUT" if strong else "BREAKOUT" if base else "PULLBACK" if pull else "NEAR_BREAKOUT" if near else None
        if not category: continue
        score=0.0
        score += {"STRONG_BREAKOUT":35,"BREAKOUT":25,"PULLBACK":22,"NEAR_BREAKOUT":15}[category]
        score += 15*clip01(1-r.conv_gap/.05) if r.conv_gap>=0 else 0
        score += 10*clip01(r.headroom448/.20)
        score += 10*clip01((r.volume_ratio-1.5)/1.5) if r.volume_ratio>=1.5 else 0
        score += 10*clip01(r.body_pct/.04) if r.body_pct>0 else 0
        score += 5*clip01((r.close_pos-.5)/.5) if pd.notna(r.close_pos) else 0
        score += 5 if r.ac>r.prior20_close_high else 0
        score += 5*clip01(np.log10(max(r.turnover20,1)/a.min_turnover+1)/np.log10(11))
        rows.append({
            "asof":r.date,"exchange":r.exchange,"series_id":r.series_id,"code":str(r.code).zfill(6),"name":r.get("name",""),
            "category":category,"score_pre_regime":round(score,2),"close":r.close,
            "ema112":r.ema112,"ema224":r.ema224,"ema448":r.ema448,"dist224_pct":r.dist224*100,
            "convergence_pct":r.conv_gap*100,"headroom448_pct":r.headroom448*100,
            "volume_ratio":r.volume_ratio,"body_pct":r.body_pct*100,"close_pos":r.close_pos,
            "upper_wick_ratio":r.upper_wick_ratio,"turnover20_krw":r.turnover20,"below80":int(r.below80),
            "breakout_age":breakout_age
        })
    latest=pd.concat(enriched,ignore_index=True) if enriched else pd.DataFrame()
    breadth={}
    if not latest.empty:
        for ex,q in latest.groupby("exchange"):
            breadth[ex]={
                "breadth20":float((q.ret20>0).mean()),"breadth60":float((q.ret60>0).mean()),"n":int(len(q))
            }
    out=pd.DataFrame(rows)
    if out.empty:
        out.to_csv(a.out/"recommendations_latest.csv",index=False,encoding="utf-8-sig")
        print(f"asof={last_date.date()} no candidates"); return
    out["market_breadth20"]=out.exchange.map(lambda x:breadth.get(x,{}).get("breadth20",np.nan))
    out["market_breadth60"]=out.exchange.map(lambda x:breadth.get(x,{}).get("breadth60",np.nan))
    out["regime"]=np.select([out.market_breadth20>=.60,out.market_breadth20>=.50],["GOOD","NORMAL"],default="WEAK")
    out["score"]=out.score_pre_regime+np.select([out.regime.eq("GOOD"),out.regime.eq("NORMAL")],[10,5],default=0)
    out=out.sort_values(["score","turnover20_krw"],ascending=[False,False]).reset_index(drop=True)
    out["rank"]=np.arange(1,len(out)+1)
    cols=["rank","asof","exchange","code","name","category","score","regime","close","ema112","ema224","ema448","dist224_pct","convergence_pct","headroom448_pct","volume_ratio","body_pct","close_pos","upper_wick_ratio","turnover20_krw","below80","breakout_age","market_breadth20","market_breadth60","series_id"]
    out[cols].to_csv(a.out/"recommendations_latest.csv",index=False,encoding="utf-8-sig")
    print(f"asof={last_date.date()} candidates={len(out)}")
    print("market breadth:",breadth)
    show=out[cols].head(a.top).copy()
    for c in ["market_breadth20","market_breadth60"]:show[c]=(show[c]*100).round(1)
    print(show.to_string(index=False))
if __name__=="__main__":main()
