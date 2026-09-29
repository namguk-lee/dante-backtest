#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Historical validation for transparent/public Dante technique approximations.

Research only:
- no proprietary indicator claims
- no buy/sell calls
- no parameter tuning against the result
- one event per technique/state after a 20-session cooldown
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from dante_signal_matrix_research import load, features

TECHS=("256_LONG","MA_HIT_112_224","MA_HIT_224_448","BOWL3_STRUCTURE","PUBLIC_CONFLUENCE_2PLUS")

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--ko",type=Path,required=True)
    p.add_argument("--kq",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("dante_public_backtest"))
    p.add_argument("--min-turnover",type=float,default=5_000_000_000)
    p.add_argument("--cooldown",type=int,default=20)
    return p.parse_args()

def days_since(mask):
    arr=np.full(len(mask),np.nan)
    last=-10**9
    for i,v in enumerate(mask.fillna(False).to_numpy()):
        if v:last=i
        if last>-10**8:arr[i]=i-last
    return arr

def bowl_mask(z):
    out=np.zeros(len(z),dtype=bool)
    if len(z)<520:return pd.Series(out,index=z.index)
    dist224=(z.ac/z.ema224-1)*100
    pre=(z.ac>=z.ema112)&(z.ema112_slope20>0)&(dist224.between(-8,15))&z.ema448.notna()
    idx=np.flatnonzero(pre.fillna(False).to_numpy())
    for i in idx:
        if i<519:continue
        start=max(0,i-219)
        tail=z.iloc[start:i+1]
        trough_i=int(tail.ac.idxmin())
        if trough_i<=0:continue
        peak_start=max(0,trough_i-160)
        prepeak=z.iloc[peak_start:trough_i+1]
        if len(prepeak)<15:continue
        peak_i=int(prepeak.ac.idxmax())
        decline_days=trough_i-peak_i
        base_days=i-trough_i
        peak=float(z.at[peak_i,"ac"]); trough=float(z.at[trough_i,"ac"])
        decline_pct=(trough/peak-1)*100 if peak>0 else np.nan
        if decline_days>=10 and pd.notna(decline_pct) and decline_pct<=-20 and base_days>=40 and base_days>=decline_days:
            out[i]=True
    return pd.Series(out,index=z.index)

def sparse_events(mask,cooldown):
    ids=[]; last=-10**9
    prev=False
    for i,v in enumerate(mask.fillna(False).to_numpy()):
        if v and not prev and i-last>=cooldown:
            ids.append(i); last=i
        prev=bool(v)
    return ids

def split_name(dt):
    y=pd.Timestamp(dt).year
    if 2021<=y<=2023:return "TRAIN_2021_2023"
    if 2024<=y<=2025:return "VALID_2024_2025"
    if y==2026:return "TEST_2026"
    return "OTHER"

def add_event(rows,z,i,tech):
    cur=z.iloc[i]
    rec={
        "date":cur.date,"code":str(cur.code).zfill(6),"name":cur.get("name",""),
        "exchange":cur.exchange,"technique":tech,"split":split_name(cur.date),
        "close":float(cur.close),"amount20":float(cur.amount20) if pd.notna(cur.amount20) else np.nan,
        "ema112_slope20_pct":float(cur.ema112_slope20) if pd.notna(cur.ema112_slope20) else np.nan,
        "ema224_slope20_pct":float(cur.ema224_slope20) if pd.notna(cur.ema224_slope20) else np.nan,
        "ema448_slope20_pct":float(cur.ema448_slope20) if pd.notna(cur.ema448_slope20) else np.nan,
        "ma_turn_quality":"TURNED_UP" if pd.notna(cur.ema112_slope20) and cur.ema112_slope20>0 else "NOT_TURNED",
        "current_volume_ratio20":float(cur.vr20) if pd.notna(cur.vr20) else np.nan,
        "max_volume_ratio20_last20":float(z.iloc[max(0,i-19):i+1].vr20.max()) if z.iloc[max(0,i-19):i+1].vr20.notna().any() else np.nan,
        "accum_cool":bool(
            pd.notna(cur.vr20) and cur.vr20<=1.2 and
            z.iloc[max(0,i-19):i+1].vr20.notna().any() and
            float(z.iloc[max(0,i-19):i+1].vr20.max())>=2.0
        ),
        "dist112_pct":float((cur.ac/cur.ema112-1)*100) if pd.notna(cur.ema112) else np.nan,
        "dist224_pct":float((cur.ac/cur.ema224-1)*100) if pd.notna(cur.ema224) else np.nan,
        "dist448_pct":float((cur.ac/cur.ema448-1)*100) if pd.notna(cur.ema448) else np.nan,
    }
    entry=float(cur.ac)
    for h in (5,20,60):
        j=i+h
        rec[f"ret{h}_pct"]=(float(z.at[j,"ac"])/entry-1)*100 if j<len(z) else np.nan
        rec[f"ret{h}_net50bp_pct"]=rec[f"ret{h}_pct"]-.5 if pd.notna(rec[f"ret{h}_pct"]) else np.nan
        end=min(len(z)-1,j)
        if i+1<=end:
            path=z.iloc[i+1:end+1]
            rec[f"mae{h}_pct"]=(float(path.al.min())/entry-1)*100
            rec[f"mfe{h}_pct"]=(float(path.ah.max())/entry-1)*100
        else:
            rec[f"mae{h}_pct"]=np.nan; rec[f"mfe{h}_pct"]=np.nan
    rows.append(rec)

def per_series(g,min_turnover,cooldown):
    z=features(g)
    if len(z)<520:return []
    liq=z.amount20>=min_turnover

    c5=(z.ema5>z.ema112)&(z.ema5.shift(1)<=z.ema112.shift(1))
    p112=(z.ac>=z.ema112)&(z.ac.shift(1)<z.ema112.shift(1))
    p224=(z.ac>=z.ema224)&(z.ac.shift(1)<z.ema224.shift(1))
    d5=days_since(c5); d112=days_since(p112); d224=days_since(p224)
    reverse=(z.ema112<z.ema224)&(z.ema224<z.ema448)

    sig256=liq&(d5<=20)&(z.ema5>z.ema112)&(z.ac>=z.ema112)&(z.ac<z.ema224)
    sig112=liq&reverse&(d112<=60)&(z.ac>=z.ema112)&(z.ac<z.ema224)
    sig224=liq&(d224<=60)&(z.ac>=z.ema224)&(z.ac<z.ema448)
    sigbowl=liq&bowl_mask(z)

    masks={
        "256_LONG":sig256,
        "MA_HIT_112_224":sig112,
        "MA_HIT_224_448":sig224,
        "BOWL3_STRUCTURE":sigbowl,
    }
    stack=pd.DataFrame({k:v.astype(int) for k,v in masks.items()})
    masks["PUBLIC_CONFLUENCE_2PLUS"]=liq&(stack.sum(axis=1)>=2)

    rows=[]
    for tech,mask in masks.items():
        for i in sparse_events(mask,cooldown):
            if pd.Timestamp(z.at[i,"date"]).year<2021:continue
            add_event(rows,z,i,tech)
    return rows

def summarize(events):
    rows=[]
    groups=[]
    for tech in TECHS:
        for split in ("TRAIN_2021_2023","VALID_2024_2025","TEST_2026","ALL"):
            for turn in ("ALL","TURNED_UP","NOT_TURNED"):
                groups.append((tech,split,turn))
    for tech,split,turn in groups:
        q=events[events.technique.eq(tech)]
        if split!="ALL":q=q[q.split.eq(split)]
        if turn!="ALL":q=q[q.ma_turn_quality.eq(turn)]
        if q.empty:continue
        rec={"technique":tech,"split":split,"turn_quality":turn,"events":len(q)}
        for h in (5,20,60):
            v=pd.to_numeric(q[f"ret{h}_net50bp_pct"],errors="coerce").dropna()
            rec[f"n{h}"]=len(v)
            rec[f"mean{h}_net_pct"]=float(v.mean()) if len(v) else np.nan
            rec[f"median{h}_net_pct"]=float(v.median()) if len(v) else np.nan
            rec[f"win{h}_pct"]=float((v>0).mean()*100) if len(v) else np.nan
            mae=pd.to_numeric(q.loc[v.index,f"mae{h}_pct"],errors="coerce").dropna() if len(v) else pd.Series(dtype=float)
            mfe=pd.to_numeric(q.loc[v.index,f"mfe{h}_pct"],errors="coerce").dropna() if len(v) else pd.Series(dtype=float)
            rec[f"median_mae{h}_pct"]=float(mae.median()) if len(mae) else np.nan
            rec[f"median_mfe{h}_pct"]=float(mfe.median()) if len(mfe) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)

def summarize_evidence_filter(events):
    """Primary non-mined filter from the dated official-example study.

    Evidence observed before this backtest:
    - >=2x volume burst within prior 20 sessions
    - quieter signal/current session (<=1.2x 20d average)
    """
    rows=[]
    variants=(
        ("BASE",pd.Series(True,index=events.index)),
        ("TURNED_UP",events.ma_turn_quality.eq("TURNED_UP")),
        ("ACCUM_COOL",events.accum_cool.eq(True)),
        ("TURNED_UP_ACCUM_COOL",events.ma_turn_quality.eq("TURNED_UP")&events.accum_cool.eq(True)),
    )
    for tech in TECHS:
        for split in ("TRAIN_2021_2023","VALID_2024_2025","TEST_2026","ALL"):
            base=events[events.technique.eq(tech)]
            if split!="ALL":base=base[base.split.eq(split)]
            for variant,mask in variants:
                q=base[mask.reindex(base.index,fill_value=False)]
                if q.empty:continue
                rec={"technique":tech,"split":split,"variant":variant,"events":len(q)}
                for h in (5,20,60):
                    v=pd.to_numeric(q[f"ret{h}_net50bp_pct"],errors="coerce").dropna()
                    rec[f"n{h}"]=len(v)
                    rec[f"mean{h}_net_pct"]=float(v.mean()) if len(v) else np.nan
                    rec[f"median{h}_net_pct"]=float(v.median()) if len(v) else np.nan
                    rec[f"win{h}_pct"]=float((v>0).mean()*100) if len(v) else np.nan
                    mae=pd.to_numeric(q.loc[v.index,f"mae{h}_pct"],errors="coerce").dropna() if len(v) else pd.Series(dtype=float)
                    mfe=pd.to_numeric(q.loc[v.index,f"mfe{h}_pct"],errors="coerce").dropna() if len(v) else pd.Series(dtype=float)
                    rec[f"median_mae{h}_pct"]=float(mae.median()) if len(mae) else np.nan
                    rec[f"median_mfe{h}_pct"]=float(mfe.median()) if len(mfe) else np.nan
                rows.append(rec)
    return pd.DataFrame(rows)

def main():
    a=args(); a.out.mkdir(parents=True,exist_ok=True)
    panel=pd.concat([load(a.ko,"KO"),load(a.kq,"KQ")],ignore_index=True)
    rows=[]
    for n,(sid,g) in enumerate(panel.groupby("series_id",sort=False),1):
        rows.extend(per_series(g,a.min_turnover,a.cooldown))
        if n%500==0:print(f"processed_series={n} events={len(rows)}",flush=True)
    events=pd.DataFrame(rows)
    events.to_csv(a.out/"public_technique_events.csv",index=False,encoding="utf-8-sig")
    if events.empty:
        print("no events"); return
    summary=summarize(events)
    summary.to_csv(a.out/"public_technique_summary.csv",index=False,encoding="utf-8-sig")
    filt=summarize_evidence_filter(events)
    filt.to_csv(a.out/"public_technique_filter_summary.csv",index=False,encoding="utf-8-sig")
    print(f"events={len(events)} series={events.code.nunique()}")
    show=summary[summary.turn_quality.eq("ALL")]
    print(show.to_string(index=False))
    print("\n=== PRE-REGISTERED EVIDENCE FILTER ===")
    key=filt[filt.variant.isin(["BASE","TURNED_UP_ACCUM_COOL"])]
    print(key.to_string(index=False))

if __name__=="__main__":
    main()
