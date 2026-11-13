#!/usr/bin/env python3
"""Parallel PUBLIC-TECHNIQUE research signals; separate from original Gongguri.

Do not merge these rows into Classic / event-first production. 256, MA-hit and
3-day rebreakout are non-identical structures. Numeric rules in this module are
transparent HYPOTHESES, not reverse-engineered proprietary indicators.

Market: 2016..2026 daily KOSPI/KOSDAQ with delisted episodes. Signals are
computed only from bars known at each signal close. Forward outcome is from
next adjusted open to day +20 adjusted close, less 0.4% round-trip fee; NOT
a stop/target backtest or a return prediction. No orders are placed.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

TYPES=("B_256_EMA5_112","B_FIRST_EMA112","B_PRE224_APPROACH","C_HILL_REBREAK_PROXY")
START=pd.Timestamp("2018-01-01")
WSTART=pd.Timestamp("2026-11-09")
WEND=pd.Timestamp("2026-11-13")
COOLDOWN=40
HIST_MIN=520

def features(g):
    z=g.sort_values("date").reset_index(drop=True).copy()
    n=len(z)
    adj=z.adjusted_close.to_numpy(dtype=float)
    raw=z.close.to_numpy(dtype=float)
    fac=np.divide(adj,raw,out=np.ones(n),where=raw>0)
    z["ao"]=z.open.to_numpy(dtype=float)*fac
    z["ah"]=z.high.to_numpy(dtype=float)*fac
    z["al"]=z.low.to_numpy(dtype=float)*fac
    z["ac"]=adj
    z["e3"]=z.ac.ewm(span=3,adjust=False,min_periods=3).mean()
    for k in (5,20,60,112,224,448):
        z[f"e{k}"]=z.ac.ewm(span=k,adjust=False,min_periods=k).mean()
    z["v20"]=z.volume.rolling(20).mean()
    z["vpre20"]=z.volume.shift(1).rolling(20).mean()
    z["vr"]=z.volume/z.vpre20 # pre-event denominator prevents self-inflated volume ratio
    z["turnover20"]=z.amount.shift(1).rolling(20).mean()
    z["hi20"]=z.ah.shift(1).rolling(20).max()
    z["hi3"]=z.ah.shift(1).rolling(3).max()
    z["prev_hi3"]=z.ah.shift(2).rolling(3).max()
    z["upper"]=((z.ah-np.maximum(z.ac,z.ao))/(z.ah-z.al).replace(0,np.nan)).clip(lower=0,upper=1)
    z["close_pos"]=((z.ac-z.al)/(z.ah-z.al).replace(0,np.nan)).clip(lower=0,upper=1)
    z["body"]=z.ac/z.ao-1
    z["r1"]=z.ac.pct_change(fill_method=None)
    z["e224slope20"]=z.e224/z.e224.shift(20)-1
    z["below80"]=(z.ac<z.e224).shift(1).rolling(80).sum()
    return z

def scan_one(g, min_turnover, recent_only=False):
    z=features(g)
    if len(z)<HIST_MIN+25:return [],[]
    c=z.ac.to_numpy(float);o=z.ao.to_numpy(float);l=z.al.to_numpy(float)
    hi=z.ah.to_numpy(float)
    e3=z.e3.to_numpy(float);e5=z.e5.to_numpy(float)
    e112=z.e112.to_numpy(float);e224=z.e224.to_numpy(float);e448=z.e448.to_numpy(float)
    volume=z.volume.to_numpy(float)
    v20=z.vpre20.to_numpy(float)
    turnover=z.turnover20.to_numpy(float)
    rv=z.vr.to_numpy(float)
    hi20=z.hi20.to_numpy(float)
    hi3=z.hi3.to_numpy(float)
    prev_hi3=z.prev_hi3.to_numpy(float)
    below=z.below80.to_numpy(float)
    body=z.body.to_numpy(float)
    cp=z.close_pos.to_numpy(float)
    upper=z.upper.to_numpy(float)
    dates=pd.to_datetime(z.date)
    last={typ:-100000 for typ in TYPES}
    ev=[];coverage=[]
    for i in range(HIST_MIN,len(z)):
        day=dates.iat[i]
        if day<START:continue
        inweek=WSTART<=day<=WEND
        if inweek:coverage.append((str(day.date()),str(z.code.iloc[i]).zfill(6)))
        if recent_only and not inweek:continue
        vals=(o[i],c[i],hi[i],l[i],e112[i],e224[i],e448[i],rv[i],turnover[i],below[i],v20[i])
        if not all(np.isfinite(v) and v>0 for v in vals):continue
        if turnover[i]<min_turnover:continue
        reverse=e112[i]<e224[i]<e448[i]
        longbelow=below[i]>=48
        detected=[]
        # 256 medium-term variant: EMA5 crosses EMA112 while EMA224 overhead.
        if (reverse and longbelow and e5[i]>e112[i] and e5[i-1]<=e112[i-1] and
                e112[i]<=c[i]<e224[i] and rv[i]>=1.15):
            detected.append("B_256_EMA5_112")
        # Price newly regains the 112-day EMA, separate from MA cross / 224 break.
        if (reverse and longbelow and c[i]>=e112[i] and c[i-1]<e112[i-1] and
                c[i]<e224[i] and rv[i]>=1.4 and body[i]>=.01):
            detected.append("B_FIRST_EMA112")
        # First entry into a narrower pre-224 zone from below: no 224 cross yet.
        dist=c[i]/e224[i]-1
        pre_dist=c[i-1]/e224[i-1]-1
        if (longbelow and -.06<=dist<0 and pre_dist<-.06 and
                e112[i]<=c[i] and rv[i]>=1.1):
            detected.append("B_PRE224_APPROACH")
        # 3-day 'hill' is qualitative in official teaching: this is an explicitly
        # named rebreakout PROXY requiring prior true volume anchor + lower-volume
        # correction + new 3-bar high close. Not equivalent to official hill.
        if (c[i]>e224[i] and c[i]>e112[i] and c[i]>hi3[i] and
                c[i-1]<=prev_hi3[i] and rv[i]>=1.15):
            for a in range(max(HIST_MIN,i-25),i-5):
                if not all(np.isfinite(v) and v>0 for v in (v20[a],c[a],hi20[a],o[a])):
                    continue
                # prior strong closing candle and volume -> nearby support zone
                if not (rv[a]>=1.8 and body[a]>=.025 and cp[a]>=.65 and c[a]>hi20[a]):
                    continue
                if c[i]<o[a]:continue
                pull=c[a+2:i]
                pv=volume[a+2:i]
                if len(pull)<2:continue
                pull_min=float(np.nanmin(l[a+2:i]))
                pull_depth=pull_min/float(hi[a])-1
                if not(-.22<=pull_depth<=-.025):continue
                if float(np.nanmedian(pv))>volume[a]*.75:continue
                if float(np.nanmin(c[a+1:i]))<o[a]*.98:continue
                detected.append("C_HILL_REBREAK_PROXY")
                break
        for typ in detected:
            if i-last[typ]<COOLDOWN:continue
            last[typ]=i
            anchor=min(i-25,HIST_MIN)
            adjfac=c[i]/float(z.close.iloc[i])
            record=dict(
                date=str(day.date()),code=str(z.code.iloc[i]).zfill(6),
                name=str(z.name.iloc[i]),exchange=str(z.exchange.iloc[i]),
                series_id=str(z.series_id.iloc[i]),signal=typ,
                category="C_NEW_REBREAK_PROXY" if typ.startswith("C_") else "B_EARLY_PUBLIC",
                price=float(z.close.iloc[i]),turnover20=int(turnover[i]),
                volratio=float(rv[i]),dist224_pct=float(dist*100),
                closepos_pct=float(cp[i]*100),upperwick_pct=float(upper[i]*100),
                candle_body_pct=float(body[i]*100),
                e112=float(e112[i]/adjfac),e224=float(e224[i]/adjfac),
                e448=float(e448[i]/adjfac),
                e224slope20_pct=float(z.e224slope20.iloc[i]*100),
                below224_prior80=int(below[i]),
                signal_support_reference=float(min(o[i],e112[i])/adjfac),
                old_relation="TO_COMPARE_WITH_ORIGINAL",
                evidence="OFFICIAL_STRUCTURE+RESEARCH_NUMERIC_PROXY")
            j=i+20
            record["mature20"]=j<len(z)
            if j<len(z) and o[i+1]>0 and c[j]>0:
                record["next_open_20close_net_pct"]=float((c[j]/o[i+1]-1)*100-.4)
                record["mae20_pct"]=float((np.nanmin(l[i+1:j+1])/o[i+1]-1)*100)
            else:
                record["next_open_20close_net_pct"]=np.nan
                record["mae20_pct"]=np.nan
            ev.append(record)
    return ev,coverage

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ko",type=Path,required=True)
    ap.add_argument("--kq",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--min-turnover",type=float,default=5_000_000_000)
    args=ap.parse_args()
    from dante_classic_scanner import load
    market=pd.concat([load(args.ko,"KO"),load(args.kq,"KQ")],ignore_index=True)
    args.out.mkdir(exist_ok=True,parents=True)
    all_ev=[];coverage=[]
    for n,(sid,g) in enumerate(market.groupby("series_id",sort=False),1):
        ev,cv=scan_one(g,args.min_turnover)
        all_ev.extend(ev);coverage.extend(cv)
        if n%300==0:print("processed_episodes",n,"events",len(all_ev),flush=True)
    df=pd.DataFrame(all_ev)
    if df.empty:raise RuntimeError("No events detected")
    df["split"]=np.select([df.date.str[:4].astype(int)<=2021,df.date.str[:4].astype(int)<=2024],
                           ["TRAIN","VALID"],default="TEST")
    df.sort_values(["date","signal","code"]).to_csv(args.out/"all_2018_2026.csv",index=False,encoding="utf-8-sig")
    w=df[df.date.between("2026-11-09","2026-11-13")].copy()
    w.sort_values(["date","signal","volratio"],ascending=[True,True,False]).to_csv(
        args.out/"new_public_research_week.csv",index=False,encoding="utf-8-sig")
    (w.groupby(["date","signal"]).size().unstack(fill_value=0).to_csv(
        args.out/"weekly_signal_counts.csv",encoding="utf-8-sig"))
    cov=pd.DataFrame(coverage,columns=["date","code"])
    cov.groupby("date").code.nunique().to_csv(args.out/"coverage.csv",encoding="utf-8-sig")
    grouped=[]
    for (typ,split),q in df.groupby(["signal","split"]):
        matured=q.dropna(subset=["next_open_20close_net_pct"])
        ret=matured.next_open_20close_net_pct
        grouped.append(dict(signal=typ,split=split,signals=len(q),mature=len(matured),
           mean20=float(ret.mean()),median20=float(ret.median()),
           win20_pct=float(100*(ret>0).mean()),mean_mae20_pct=float(matured.mae20_pct.mean()),
           year2025=int((q.date.str[:4]=="2025").sum()),
           year2026=int((q.date.str[:4]=="2026").sum())))
    summary=pd.DataFrame(grouped)
    summary.to_csv(args.out/"historical_20d_summary.csv",index=False,encoding="utf-8-sig")
    # Distinct latest B/C signal codes' FULL actual OHLCV for chart rendering.
    chosen=[]
    if not w.empty:
        seen=set()
        for _,r in w.sort_values(["volratio","turnover20"],ascending=False).iterrows():
            if r.code in seen:continue
            seen.add(r.code);chosen.append(r.code)
            if len(chosen)>=22:break
    x=market[market.code.isin(chosen)&market.date.between("2026-01-01","2026-11-13")].copy()
    x.to_csv(args.out/"selected_new_ohlcv.csv",index=False,encoding="utf-8-sig")
    print("PANEL_TO",market.date.max(),"TOTAL_EVENTS",len(df),"WEEK_NEW",len(w),flush=True)
    print("WEEK COUNTS\n",w.groupby(["date","signal"]).size().to_string(),flush=True)
    print("HISTORICAL (UNOPTIMIZED)\n",summary.to_string(index=False),flush=True)
    print("CAUTION: old and new criteria may overlap; see code/date crosswalk.",flush=True)

if __name__=="__main__":
    main()
