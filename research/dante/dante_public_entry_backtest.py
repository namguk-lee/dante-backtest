#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Delayed-entry study for public Dante technique approximations.

Research only. Separates:
1) setup discovery (256 / MA-hit / Bowl-3)
2) later pullback/settlement entry proxy

The entry proxy is NOT claimed to be Dante's proprietary/exact formula.
Primary tolerance=5%; 3% and 8% are sensitivity checks, not optimization.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from dante_signal_matrix_research import load, features
from dante_public_technique_backtest import days_since, bowl_mask, sparse_events, split_name

TECHS=("256_LONG","MA_HIT_112_224","MA_HIT_224_448","BOWL3_STRUCTURE")
TOLS=(0.03,0.05,0.08)

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--ko",type=Path,required=True)
    p.add_argument("--kq",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("dante_entry_backtest"))
    p.add_argument("--min-turnover",type=float,default=5_000_000_000)
    p.add_argument("--cooldown",type=int,default=20)
    p.add_argument("--entry-window",type=int,default=20)
    return p.parse_args()

def setup_masks(z,min_turnover):
    liq=z.amount20>=min_turnover
    c5=(z.ema5>z.ema112)&(z.ema5.shift(1)<=z.ema112.shift(1))
    p112=(z.ac>=z.ema112)&(z.ac.shift(1)<z.ema112.shift(1))
    p224=(z.ac>=z.ema224)&(z.ac.shift(1)<z.ema224.shift(1))
    d5=days_since(c5); d112=days_since(p112); d224=days_since(p224)
    reverse=(z.ema112<z.ema224)&(z.ema224<z.ema448)
    return {
        "256_LONG":liq&(d5<=20)&(z.ema5>z.ema112)&(z.ac>=z.ema112)&(z.ac<z.ema224),
        "MA_HIT_112_224":liq&reverse&(d112<=60)&(z.ac>=z.ema112)&(z.ac<z.ema224),
        "MA_HIT_224_448":liq&(d224<=60)&(z.ac>=z.ema224)&(z.ac<z.ema448),
        "BOWL3_STRUCTURE":liq&bowl_mask(z),
    }

def ma_pair(tech):
    if tech in ("256_LONG","MA_HIT_112_224"):
        return 112,224
    return 224,448

def volume_context(z,i):
    w=z.iloc[max(0,i-19):i+1]
    maxvr=float(w.vr20.max()) if w.vr20.notna().any() else np.nan
    curvr=float(z.at[i,"vr20"]) if pd.notna(z.at[i,"vr20"]) else np.nan
    return maxvr,curvr

def first_pullback_entry(z,setup_i,tech,tol,window):
    ref_n,target_n=ma_pair(tech)
    end=min(len(z)-1,setup_i+window)
    for j in range(setup_i+1,end+1):
        ref=z.at[j,f"ema{ref_n}"]; target=z.at[j,f"ema{target_n}"]
        if pd.isna(ref) or pd.isna(target) or ref<=0:continue
        close=float(z.at[j,"ac"]); low=float(z.at[j,"al"])
        # "settlement": majority of recent closes remain on/above the reference MA.
        s=max(0,j-4)
        recent=z.iloc[s:j+1]
        settle=(recent.ac>=recent[f"ema{ref_n}"]).sum()>=min(3,len(recent))
        # First quiet pullback close near the reclaimed reference MA.
        near=(low<=float(ref)*(1+tol)) and (close>=float(ref)) and (close<=float(ref)*(1+2*tol))
        below_target=close<float(target)
        quiet=pd.notna(z.at[j,"vr20"]) and float(z.at[j,"vr20"])<=1.2
        if settle and near and below_target and quiet:
            maxvr,curvr=volume_context(z,j)
            return {
                "entry_i":j,"entry_date":z.at[j,"date"],"entry_lag":j-setup_i,
                "ref_ma":ref_n,"target_ma":target_n,
                "entry_dist_ref_pct":(close/float(ref)-1)*100,
                "target_headroom_pct":(float(target)/close-1)*100,
                "entry_volume_ratio20":curvr,
                "max_volume_ratio20_last20":maxvr,
                "accum_cool":bool(pd.notna(maxvr) and maxvr>=2.0 and pd.notna(curvr) and curvr<=1.2),
            }
    return None

def add_returns(rec,z,entry_i):
    entry=float(z.at[entry_i,"ac"])
    for h in (5,20,60):
        k=entry_i+h
        rec[f"ret{h}_net50bp_pct"]=(float(z.at[k,"ac"])/entry-1)*100-.5 if k<len(z) else np.nan
        end=min(len(z)-1,k)
        if entry_i+1<=end:
            path=z.iloc[entry_i+1:end+1]
            rec[f"mae{h}_pct"]=(float(path.al.min())/entry-1)*100
            rec[f"mfe{h}_pct"]=(float(path.ah.max())/entry-1)*100
        else:
            rec[f"mae{h}_pct"]=np.nan; rec[f"mfe{h}_pct"]=np.nan
    return rec

def per_series(g,min_turnover,cooldown,window):
    z=features(g)
    if len(z)<520:return []
    masks=setup_masks(z,min_turnover)
    rows=[]
    for tech,mask in masks.items():
        for setup_i in sparse_events(mask,cooldown):
            if pd.Timestamp(z.at[setup_i,"date"]).year<2021:continue
            for tol in TOLS:
                ent=first_pullback_entry(z,setup_i,tech,tol,window)
                if not ent:continue
                i=ent["entry_i"]
                rec={
                    "setup_date":z.at[setup_i,"date"],"entry_date":ent["entry_date"],
                    "code":str(z.at[i,"code"]).zfill(6),"name":z.at[i].get("name",""),
                    "exchange":z.at[i,"exchange"],"technique":tech,"split":split_name(z.at[i,"date"]),
                    "tolerance_pct":tol*100,
                    "entry_lag":ent["entry_lag"],"ref_ma":ent["ref_ma"],"target_ma":ent["target_ma"],
                    "entry_close":float(z.at[i,"close"]),
                    "entry_dist_ref_pct":ent["entry_dist_ref_pct"],
                    "target_headroom_pct":ent["target_headroom_pct"],
                    "entry_volume_ratio20":ent["entry_volume_ratio20"],
                    "max_volume_ratio20_last20":ent["max_volume_ratio20_last20"],
                    "accum_cool":ent["accum_cool"],
                    "ema112_slope20_pct":float(z.at[i,"ema112_slope20"]) if pd.notna(z.at[i,"ema112_slope20"]) else np.nan,
                    "ema224_slope20_pct":float(z.at[i,"ema224_slope20"]) if pd.notna(z.at[i,"ema224_slope20"]) else np.nan,
                    "ema448_slope20_pct":float(z.at[i,"ema448_slope20"]) if pd.notna(z.at[i,"ema448_slope20"]) else np.nan,
                }
                rec=add_returns(rec,z,i)
                rows.append(rec)
    return rows

def summarize(events):
    rows=[]
    for tech in TECHS:
        for tol in (3.0,5.0,8.0):
            for split in ("TRAIN_2021_2023","VALID_2024_2025","TEST_2026","ALL"):
                base=events[(events.technique.eq(tech))&(events.tolerance_pct.eq(tol))]
                if split!="ALL":base=base[base.split.eq(split)]
                for variant,mask in (
                    ("PULLBACK",pd.Series(True,index=base.index)),
                    ("PULLBACK_ACCUM_COOL",base.accum_cool.eq(True)),
                    ("PULLBACK_TURNED",pd.to_numeric(base.ema112_slope20_pct,errors="coerce")>0),
                    ("PULLBACK_TURNED_ACCUM",
                     (pd.to_numeric(base.ema112_slope20_pct,errors="coerce")>0)&base.accum_cool.eq(True)),
                ):
                    q=base[mask.fillna(False)]
                    if q.empty:continue
                    rec={"technique":tech,"tolerance_pct":tol,"split":split,"variant":variant,"events":len(q)}
                    rec["median_entry_lag"]=float(pd.to_numeric(q.entry_lag,errors="coerce").median())
                    rec["median_target_headroom_pct"]=float(pd.to_numeric(q.target_headroom_pct,errors="coerce").median())
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
        rows.extend(per_series(g,a.min_turnover,a.cooldown,a.entry_window))
        if n%500==0:print(f"processed_series={n} entries={len(rows)}",flush=True)
    events=pd.DataFrame(rows)
    events.to_csv(a.out/"pullback_entry_events.csv",index=False,encoding="utf-8-sig")
    if events.empty:
        print("no entries"); return
    summary=summarize(events)
    summary.to_csv(a.out/"pullback_entry_summary.csv",index=False,encoding="utf-8-sig")
    print(f"entries={len(events)} series={events.code.nunique()}")
    primary=summary[(summary.tolerance_pct.eq(5.0))&summary.variant.isin(["PULLBACK","PULLBACK_ACCUM_COOL"])]
    print("\n=== PRIMARY 5% PULLBACK ENTRY ===")
    print(primary.to_string(index=False))
    print("\n=== SENSITIVITY: 3/5/8% PULLBACK ===")
    sens=summary[(summary.variant.eq("PULLBACK_ACCUM_COOL"))&summary.split.isin(["VALID_2024_2025","TEST_2026","ALL"])]
    print(sens.to_string(index=False))

if __name__=="__main__":
    main()
