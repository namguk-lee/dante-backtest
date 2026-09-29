#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DANTE_CLASSIC_V1 scanner.

Transparent approximation of the publicly described Dante chart structure:
decline -> longer base -> EMA224 recovery/approach, anchor candle, prior-hill
breakout/support ("concrete"), 112 recovery, and structural targets.

Private/proprietary Dante indicators are intentionally not reproduced.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--ko",type=Path,required=True)
    p.add_argument("--kq",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("dante_classic_results"))
    p.add_argument("--max-charts",type=int,default=8)
    p.add_argument("--min-turnover",type=float,default=5_000_000_000)
    return p.parse_args()


def load(path,ex):
    x=pd.read_parquet(path).copy()
    x["date"]=pd.to_datetime(x["date"],errors="coerce")
    x["code"]=x["code"].astype(str).str.replace(r"\.0$","",regex=True).str.zfill(6)
    if "exchange" not in x:x["exchange"]=ex
    if "name" not in x:x["name"]=""
    if "adjusted_close" not in x:x["adjusted_close"]=x["close"]
    if "amount" not in x:x["amount"]=x["close"]*x["volume"]
    return x.sort_values(["series_id","date"])


def feat(g):
    g=g.sort_values("date").copy().reset_index(drop=True)
    factor=(g.adjusted_close/g.close).replace([np.inf,-np.inf],np.nan).ffill().bfill().fillna(1.0)
    g["ao"]=g.open*factor; g["ah"]=g.high*factor; g["al"]=g.low*factor; g["ac"]=g.adjusted_close
    for n in (5,15,33,56,112,224,448):
        g[f"ema{n}"]=g.ac.ewm(span=n,adjust=False,min_periods=n).mean()
    g["v20"]=g.volume.rolling(20,min_periods=20).mean()
    g["amt20"]=g.amount.rolling(20,min_periods=20).mean()
    g["vr"]=g.volume/g.v20
    g["ret1"]=g.ac.pct_change(fill_method=None)
    g["body"]=(g.ac-g.ao)/g.ao
    rng=(g.ah-g.al).replace(0,np.nan)
    g["close_pos"]=(g.ac-g.al)/rng
    g["upper_wick"]=(g.ah-np.maximum(g.ao,g.ac))/rng
    g["ema112_slope20"]=g.ema112/g.ema112.shift(20)-1
    g["ema224_slope20"]=g.ema224/g.ema224.shift(20)-1
    g["cross112"]=(g.ac>g.ema112)&(g.ac.shift(1)<=g.ema112.shift(1))
    g["cross224"]=(g.ac>g.ema224)&(g.ac.shift(1)<=g.ema224.shift(1))
    g["prior20_high"]=g.ah.shift(1).rolling(20,min_periods=20).max()
    return g


def consecutive_below_before(z,idx,line="ema224"):
    n=0
    for j in range(idx-1,-1,-1):
        if pd.isna(z.at[j,line]) or not(float(z.at[j,"ac"])<float(z.at[j,line])):break
        n+=1
    return n


def local_swing_highs(z,start,end,wing=3):
    start=max(start,wing); end=min(end,len(z)-wing)
    out=[]
    for i in range(start,end):
        h=float(z.at[i,"ah"])
        if h>=float(z.loc[i-wing:i+wing,"ah"].max()):
            out.append(i)
    return out


def find_bowl(z):
    n=len(z)
    if n<520:return None
    approach_candidates=[]
    for i in range(max(448,n-60),n):
        if pd.isna(z.at[i,"ema224"]):continue
        d=float(z.at[i,"ac"]/z.at[i,"ema224"]-1)
        if -.10<=d<=.10:
            approach_candidates.append(i)
    if not approach_candidates:return None
    approach_i=approach_candidates[0]
    below_days=consecutive_below_before(z,approach_i,"ema224")
    left=max(448,approach_i-260)
    if approach_i-left<80:return None
    b_end=max(left+30,approach_i-15)
    if b_end<=left:return None
    bottom_i=int(z.loc[left:b_end,"ac"].idxmin())
    p_start=max(448,bottom_i-140)
    p_end=bottom_i-10
    if p_end<=p_start:return None
    peak_i=int(z.loc[p_start:p_end,"ah"].idxmax())
    if peak_i>=bottom_i:return None

    decline_days=bottom_i-peak_i
    base_days=approach_i-bottom_i
    peak=float(z.at[peak_i,"ah"]); bottom=float(z.at[bottom_i,"al"])
    decline_pct=bottom/peak-1 if peak>0 else np.nan
    base=z.iloc[bottom_i:approach_i+1]
    q10=float(base.ac.quantile(.10)); q90=float(base.ac.quantile(.90))
    base_range=(q90/q10-1) if q10>0 else np.nan
    base_slope=float(base.ac.iloc[-1]/base.ac.iloc[0]-1) if len(base)>1 else np.nan
    return {
        "approach_i":approach_i,"peak_i":peak_i,"bottom_i":bottom_i,
        "below224_days_before_approach":below_days,
        "bowl1_days":decline_days,"bowl2_days":base_days,
        "bowl2_over_bowl1":base_days/decline_days if decline_days>0 else np.nan,
        "bowl1_decline_pct":decline_pct,
        "bowl2_range_pct":base_range,
        "bowl2_net_change_pct":base_slope,
    }


def find_anchor(z,bowl,min_turnover):
    bottom_i=bowl["bottom_i"]; approach_i=bowl["approach_i"]
    start=min(approach_i,max(bottom_i+10,approach_i-80))
    end=min(len(z)-1,approach_i+20)
    cand=[]
    for i in range(start,end+1):
        if any(pd.isna(z.at[i,c]) for c in ["vr","body","close_pos","amt20","ema112"]):continue
        if float(z.at[i,"amt20"])<min_turnover:continue
        if float(z.at[i,"body"])<.035 or float(z.at[i,"vr"])<1.8 or float(z.at[i,"close_pos"])<.60:
            continue
        crossed112=bool(z.at[i,"cross112"]) or float(z.at[i,"ac"])>float(z.at[i,"ema112"])
        broke20=bool(pd.notna(z.at[i,"prior20_high"]) and z.at[i,"ac"]>z.at[i,"prior20_high"])
        if not(crossed112 or broke20):continue
        strength=float(z.at[i,"body"])*min(float(z.at[i,"vr"]),8)*(1.15 if broke20 else 1)
        cand.append((strength,i,broke20))
    if not cand:return None
    _,i,broke20=max(cand)
    anchor_open=float(z.at[i,"ao"])
    post=z.iloc[i:]
    closes_below=int((post.ac<anchor_open).sum())
    return {
        "anchor_i":i,"anchor_open_adj":anchor_open,
        "anchor_ret_pct":float(z.at[i,"ret1"]),
        "anchor_body_pct":float(z.at[i,"body"]),
        "anchor_volume_ratio":float(z.at[i,"vr"]),
        "anchor_broke20":bool(broke20),
        "anchor_closes_below":closes_below,
        "anchor_alive_strict":bool(closes_below==0 and float(z.iloc[-1].ac)>=anchor_open),
    }


def find_concrete(z,anchor_i):
    swings=local_swing_highs(z,max(448,anchor_i-120),anchor_i-3,wing=3)
    if not swings:return None
    resistance_i=swings[-1]
    resistance=float(z.at[resistance_i,"ah"])
    breakout_i=None
    for j in range(anchor_i,min(anchor_i+11,len(z))):
        if float(z.at[j,"ac"])>resistance:
            breakout_i=j;break
    if breakout_i is None:return {
        "resistance_i":resistance_i,"resistance_adj":resistance,
        "concrete_breakout":False,"concrete_hold_strict":False,
        "concrete_hold_loose":False,"concrete_closes_below":np.nan,
    }
    post=z.iloc[breakout_i:]
    below=int((post.ac<resistance).sum())
    return {
        "resistance_i":resistance_i,"resistance_adj":resistance,
        "concrete_breakout":True,"concrete_breakout_i":breakout_i,
        "concrete_closes_below":below,
        "concrete_hold_strict":bool(below==0 and float(z.iloc[-1].ac)>=resistance),
        "concrete_hold_loose":bool(below<=1 and float(z.iloc[-1].ac)>=resistance),
    }


def nearest_prior_resistance(z,cur_i,cur_price):
    swings=local_swing_highs(z,max(448,cur_i-260),cur_i-5,wing=4)
    prices=[float(z.at[i,"ah"]) for i in swings if float(z.at[i,"ah"])>cur_price*1.03]
    return min(prices) if prices else np.nan


def classify_one(g,min_turnover):
    z=feat(g)
    bowl=find_bowl(z)
    if not bowl:return None,z
    anchor=find_anchor(z,bowl,min_turnover)
    if not anchor:return None,z
    concrete=find_concrete(z,anchor["anchor_i"])
    cur=z.iloc[-1]
    factor=float(cur.ac/cur.close) if float(cur.close)>0 else 1.0
    cur_price=float(cur.close)
    ema={n:(float(cur[f"ema{n}"]/factor) if pd.notna(cur[f"ema{n}"]) else np.nan) for n in (5,15,33,56,112,224,448)}
    dist224=float(cur.ac/cur.ema224-1) if pd.notna(cur.ema224) else np.nan
    dist112=float(cur.ac/cur.ema112-1) if pd.notna(cur.ema112) else np.nan

    recent112=z.iloc[max(bowl["bottom_i"],len(z)-100):]
    had112=bool(recent112.cross112.any() or (recent112.ac>recent112.ema112).any())
    ema112_rising=bool(pd.notna(cur.ema112_slope20) and cur.ema112_slope20>0)

    if -.10<=dist224<0 and had112:
        state="BOWL3_PRE224"
    elif 0<=dist224<=.10 and had112:
        state="BOWL3_224_HOLD"
    elif cur.ac>cur.ema224 and pd.notna(cur.ema448) and cur.ac<cur.ema448:
        state="POST224_TO448"
    else:
        state="STRUCTURE_ONLY"

    bowl_duration_ok=bool(bowl["bowl2_days"]>=bowl["bowl1_days"])
    four_month_ok=bool(bowl["below224_days_before_approach"]>=80)
    big_decline=bool(bowl["bowl1_decline_pct"]<=-.20)
    base_reasonable=bool(bowl["bowl2_range_pct"]<=.45)
    anchor_strict=bool(anchor["anchor_alive_strict"])
    concrete_strict=bool(concrete and concrete.get("concrete_hold_strict",False))
    concrete_loose=bool(concrete and concrete.get("concrete_hold_loose",False))
    near224=bool(-.10<=dist224<=.10)

    score=0
    score+=18 if bowl_duration_ok else 0
    score+=15 if four_month_ok else min(10,bowl["below224_days_before_approach"]/8)
    score+=8 if big_decline else 0
    score+=8 if base_reasonable else 0
    score+=15 if anchor_strict else 0
    score+=8 if anchor["anchor_broke20"] else 0
    score+=12 if concrete_strict else 7 if concrete_loose else 0
    score+=8 if had112 else 0
    score+=4 if ema112_rising else 0
    score+=4 if near224 else 0

    if all([bowl_duration_ok,four_month_ok,big_decline,anchor_strict,concrete_loose,had112,near224]):
        tier="A"
    elif all([bowl_duration_ok,four_month_ok,anchor_strict,had112,near224]):
        tier="B"
    else:
        tier="NEAR"

    prior_res=nearest_prior_resistance(z,len(z)-1,float(cur.ac))
    target_candidates=[]
    if dist224<0 and pd.notna(cur.ema224) and cur.ema224>cur.ac:
        target_candidates.append(("EMA224",float(cur.ema224/factor)))
    if pd.notna(cur.ema448) and cur.ema448>cur.ac:
        target_candidates.append(("EMA448",float(cur.ema448/factor)))
    if pd.notna(prior_res):
        target_candidates.append(("PRIOR_RESISTANCE",float(prior_res/factor)))
    target_name,target_price=(min(target_candidates,key=lambda x:x[1]) if target_candidates else ("NONE",np.nan))

    rec={
        "series_id":cur.series_id,"code":str(cur.code).zfill(6),"name":cur.get("name",""),"exchange":cur.exchange,
        "date":cur.date,"tier":tier,"state":state,"score":round(score,2),"current_close":cur_price,
        **{f"ema{n}":ema[n] for n in (5,15,33,56,112,224,448)},
        "dist112_pct":dist112*100,"dist224_pct":dist224*100,
        "ema112_slope20_pct":float(cur.ema112_slope20*100) if pd.notna(cur.ema112_slope20) else np.nan,
        "ema224_slope20_pct":float(cur.ema224_slope20*100) if pd.notna(cur.ema224_slope20) else np.nan,
        "bowl1_peak_date":z.at[bowl["peak_i"],"date"],"bowl_bottom_date":z.at[bowl["bottom_i"],"date"],
        "bowl3_approach_date":z.at[bowl["approach_i"],"date"],
        "bowl1_days":bowl["bowl1_days"],"bowl2_days":bowl["bowl2_days"],
        "bowl2_over_bowl1":bowl["bowl2_over_bowl1"],
        "bowl1_decline_pct":bowl["bowl1_decline_pct"]*100,
        "bowl2_range_pct":bowl["bowl2_range_pct"]*100,
        "below224_days_before_approach":bowl["below224_days_before_approach"],
        "bowl_duration_ok":bowl_duration_ok,"four_month_below224":four_month_ok,
        "anchor_date":z.at[anchor["anchor_i"],"date"],"anchor_open":anchor["anchor_open_adj"]/factor,
        "anchor_ret_pct":anchor["anchor_ret_pct"]*100,"anchor_body_pct":anchor["anchor_body_pct"]*100,
        "anchor_volume_ratio":anchor["anchor_volume_ratio"],"anchor_broke20":anchor["anchor_broke20"],
        "anchor_closes_below":anchor["anchor_closes_below"],"anchor_alive_strict":anchor_strict,
        "had112_recovery":had112,"ema112_rising":ema112_rising,
        "concrete_resistance":(concrete["resistance_adj"]/factor if concrete else np.nan),
        "concrete_breakout":bool(concrete and concrete.get("concrete_breakout",False)),
        "concrete_closes_below":(concrete.get("concrete_closes_below",np.nan) if concrete else np.nan),
        "concrete_hold_strict":concrete_strict,"concrete_hold_loose":concrete_loose,
        "target_type":target_name,"target_price":target_price,
        "target_upside_pct":((target_price/cur_price-1)*100 if pd.notna(target_price) and cur_price>0 else np.nan),
        "_idx_peak":bowl["peak_i"],"_idx_bottom":bowl["bottom_i"],"_idx_approach":bowl["approach_i"],
        "_idx_anchor":anchor["anchor_i"],"_idx_resistance":(concrete["resistance_i"] if concrete else np.nan),
    }
    return rec,z


def render_chart(z,rec,out_dir,bars=280):
    end=len(z); start=max(0,end-bars)
    t=z.iloc[start:].copy().reset_index().rename(columns={"index":"orig_i"})
    cur=z.iloc[-1]; factor=float(cur.ac/cur.close) if float(cur.close)>0 else 1.0
    for col in ["ao","ah","al","ac"]+[f"ema{n}" for n in (5,15,33,56,112,224,448)]:
        t[col+"_raw"]=t[col]/factor

    x=np.arange(len(t))
    fig,(ax,av)=plt.subplots(2,1,figsize=(16,9),sharex=True,gridspec_kw={"height_ratios":[4,1]})
    for i,r in t.iterrows():
        ax.vlines(i,r.al_raw,r.ah_raw,linewidth=.7)
        ax.hlines(r.ao_raw,i-.28,i,linewidth=.9)
        ax.hlines(r.ac_raw,i,i+.28,linewidth=.9)
    for n,lw in [(5,.8),(15,.8),(33,.8),(56,.8),(112,1.2),(224,1.5),(448,1.3)]:
        ax.plot(x,t[f"ema{n}_raw"],label=f"EMA{n}",linewidth=lw)

    def xpos(orig):
        hits=t.index[t.orig_i.eq(orig)]
        return int(hits[-1]) if len(hits) else None

    for label,orig,style in [
        ("Bowl1 peak",rec["_idx_peak"],"--"),("Bowl bottom",rec["_idx_bottom"],":"),
        ("Bowl3/224 area",rec["_idx_approach"],"-."),("Anchor",rec["_idx_anchor"],"--")]:
        p=xpos(int(orig))
        if p is not None:ax.axvline(p,linestyle=style,linewidth=1.0,label=label)

    for label,val in [("Anchor open",rec.get("anchor_open")),("Concrete",rec.get("concrete_resistance")),("Target",rec.get("target_price"))]:
        if pd.notna(val):
            ax.axhline(float(val),linewidth=.8,alpha=.7)
            ax.text(len(t)-1,float(val),f" {label} {float(val):,.0f}",fontsize=8,va="bottom")

    av.bar(x,t.volume)
    ax.grid(alpha=.15); av.grid(alpha=.15)
    ax.legend(loc="upper left",ncol=6,fontsize=8)
    ax.set_ylabel("Price"); av.set_ylabel("Volume")
    step=max(1,len(t)//10); ticks=list(range(0,len(t),step))
    if ticks[-1]!=len(t)-1:ticks.append(len(t)-1)
    av.set_xticks(ticks)
    av.set_xticklabels([pd.Timestamp(t.at[i,"date"]).strftime("%Y-%m-%d") for i in ticks],rotation=25,ha="right")
    ax.set_title(f'{rec["code"]} | DANTE_CLASSIC_V1 {rec["tier"]} | {rec["state"]} | score {rec["score"]:.0f}')
    note=(f'Bowl1 {rec["bowl1_days"]}d / Bowl2 {rec["bowl2_days"]}d (x{rec["bowl2_over_bowl1"]:.2f}) | '
          f'below224 {rec["below224_days_before_approach"]}d | anchor vol x{rec["anchor_volume_ratio"]:.1f}, '
          f'closes below open {rec["anchor_closes_below"]} | concrete strict={rec["concrete_hold_strict"]}')
    fig.text(.01,.01,note,fontsize=8)
    fig.tight_layout(rect=[0,.035,1,1])
    out_dir.mkdir(parents=True,exist_ok=True)
    path=out_dir/f'{rec["code"]}_DANTE_CLASSIC_V1.png'
    fig.savefig(path,dpi=150); plt.close(fig)
    return path


def main():
    a=parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    panel=pd.concat([load(a.ko,"KO"),load(a.kq,"KQ")],ignore_index=True)
    latest=panel.date.max()
    active=set(panel.loc[panel.date.eq(latest),"series_id"].astype(str))
    rows=[]; chart_inputs=[]
    total=len(active)
    for n,(sid,g) in enumerate(panel[panel.series_id.astype(str).isin(active)].groupby("series_id",sort=False),1):
        if len(g)<520:continue
        rec,z=classify_one(g,a.min_turnover)
        if rec:
            rows.append(rec); chart_inputs.append((rec,z))
        if n%250==0 or n==total:
            print(f"series {n}/{total} candidates={len(rows)}",flush=True)

    if not rows:
        pd.DataFrame().to_csv(a.out/"dante_classic_candidates.csv",index=False,encoding="utf-8-sig")
        return
    out=pd.DataFrame(rows)
    order={"A":0,"B":1,"NEAR":2}
    out["_tier_order"]=out.tier.map(order).fillna(9)
    out=out.sort_values(["_tier_order","score"],ascending=[True,False]).drop(columns=["_tier_order"]).reset_index(drop=True)
    out["rank"]=np.arange(1,len(out)+1)
    csv_cols=[c for c in out.columns if not c.startswith("_idx_")]
    out[csv_cols].to_csv(a.out/"dante_classic_candidates.csv",index=False,encoding="utf-8-sig")

    selected=out.head(a.max_charts)
    chart_map={r["series_id"]:z for r,z in chart_inputs}
    chart_dir=a.out/"charts"
    for _,r in selected.iterrows():
        render_chart(chart_map[r.series_id],r.to_dict(),chart_dir)

    print("\n=== DANTE_CLASSIC_V1 TOP ===")
    showcols=["rank","code","name","exchange","tier","state","score","current_close",
              "bowl1_days","bowl2_days","bowl2_over_bowl1","below224_days_before_approach",
              "anchor_date","anchor_volume_ratio","anchor_closes_below","concrete_hold_strict",
              "dist224_pct","ema112_slope20_pct","target_type","target_price","target_upside_pct"]
    print(out[showcols].head(30).to_string(index=False))


if __name__=="__main__":
    main()
