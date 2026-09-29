#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reverse-engineer candidate geometries for Dante's public 'share 1:1' examples.

Research only. No proprietary formula is claimed.
Each example is evaluated only with data available on/before its analysis_date.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--ko",type=Path,required=True)
    p.add_argument("--kq",type=Path,required=True)
    p.add_argument("--cases",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("dante_share_1to1_study"))
    return p.parse_args()

def load(path,ex):
    x=pd.read_parquet(path).copy()
    x["date"]=pd.to_datetime(x["date"],errors="coerce")
    x["name"]=x.get("name","").astype(str).str.strip()
    if "exchange" not in x:x["exchange"]=ex
    if "adjusted_close" not in x:x["adjusted_close"]=x["close"]
    return x.sort_values(["series_id","date"])

def feat(g,ma_n):
    z=g.sort_values("date").copy().reset_index(drop=True)
    f=(z.adjusted_close/z.close).replace([np.inf,-np.inf],np.nan).ffill().bfill().fillna(1.0)
    z["ac"]=z.adjusted_close
    z["ah"]=z.high*f
    z["al"]=z.low*f
    z["ma"]=z.ac.ewm(span=ma_n,adjust=False,min_periods=ma_n).mean()
    z["dist"]=(z.ac-z.ma)/z.ma
    z["above"]=z.ac>=z.ma
    return z

def sustained_up_crosses(z,lookback=180):
    start=max(1,len(z)-lookback)
    out=[]
    for i in range(start,len(z)-2):
        if pd.isna(z.at[i,"ma"]) or pd.isna(z.at[i-1,"ma"]):continue
        cross=(z.at[i,"ac"]>=z.at[i,"ma"]) and (z.at[i-1,"ac"]<z.at[i-1,"ma"])
        if not cross:continue
        nxt=z.iloc[i:min(i+5,len(z))]
        if len(nxt)>=3 and int((nxt.ac>=nxt.ma).sum())>=3:
            out.append(i)
    return out

def contiguous_runs(z,cross_i):
    # immediate contiguous below-MA run ending just before cross
    j=cross_i-1
    while j>=0 and pd.notna(z.at[j,"ma"]) and z.at[j,"ac"]<z.at[j,"ma"]:
        j-=1
    pre_start=j+1
    pre_end=cross_i-1
    # immediate above-MA run after cross, capped at analysis date
    k=cross_i
    while k+1<len(z) and pd.notna(z.at[k+1,"ma"]) and z.at[k+1,"ac"]>=z.at[k+1,"ma"]:
        k+=1
    return pre_start,pre_end,cross_i,k

def safe_ratio(a,b):
    if pd.isna(a) or pd.isna(b) or b==0:return np.nan
    return float(a/b)

def geometry(z,cross_i):
    ps,pe,as_,ae=contiguous_runs(z,cross_i)
    if pe<ps or ae<as_:return None
    pre=z.iloc[ps:pe+1].copy()
    post=z.iloc[as_:ae+1].copy()
    if pre.empty or post.empty:return None

    # Vertical distance candidates.
    pre_depth_max=float((-pre.dist).clip(lower=0).max())
    post_height_current=float(max(0,z.iloc[-1].dist))
    post_height_max=float(post.dist.clip(lower=0).max())
    post_height_close=float(max(0,z.at[ae,"dist"]))

    # Durations.
    pre_days=len(pre)
    post_days=len(post)

    # Integrated territory ("area") in normalized MA-distance units.
    pre_area=float((-pre.dist).clip(lower=0).sum())
    post_area=float(post.dist.clip(lower=0).sum())

    # Rectangle approximations.
    pre_rect=pre_depth_max*pre_days
    post_rect_current=post_height_current*post_days
    post_rect_max=post_height_max*post_days

    # Absolute swing around the MA using intraday extremes.
    pre_low_ratio=float(((pre.ma-pre.al)/pre.ma).clip(lower=0).max())
    post_high_ratio=float(((post.ah-post.ma)/post.ma).clip(lower=0).max())

    # Price swing anchored to MA value at the cross.
    ma0=float(z.at[cross_i,"ma"])
    pre_low=float(pre.al.min())
    post_high=float(post.ah.max())
    current=float(z.iloc[-1].ac)
    down_abs=(ma0-pre_low)/ma0 if ma0>0 else np.nan
    up_abs_max=(post_high-ma0)/ma0 if ma0>0 else np.nan
    up_abs_current=(current-ma0)/ma0 if ma0>0 else np.nan

    return {
        "cross_i":cross_i,
        "cross_date":z.at[cross_i,"date"],
        "pre_start_date":z.at[ps,"date"],
        "pre_end_date":z.at[pe,"date"],
        "post_end_date":z.at[ae,"date"],
        "pre_days":pre_days,
        "post_days":post_days,
        "pre_depth_max_pct":pre_depth_max*100,
        "post_height_current_pct":post_height_current*100,
        "post_height_max_pct":post_height_max*100,
        "post_height_end_pct":post_height_close*100,
        "pre_area":pre_area,
        "post_area":post_area,
        "pre_low_ratio_pct":pre_low_ratio*100,
        "post_high_ratio_pct":post_high_ratio*100,
        "ratio_vertical_current":safe_ratio(post_height_current,pre_depth_max),
        "ratio_vertical_max":safe_ratio(post_height_max,pre_depth_max),
        "ratio_days":safe_ratio(post_days,pre_days),
        "ratio_area":safe_ratio(post_area,pre_area),
        "ratio_rect_current":safe_ratio(post_rect_current,pre_rect),
        "ratio_rect_max":safe_ratio(post_rect_max,pre_rect),
        "ratio_intraday_extreme":safe_ratio(post_high_ratio,pre_low_ratio),
        "ratio_cross_ma_current":safe_ratio(up_abs_current,down_abs),
        "ratio_cross_ma_max":safe_ratio(up_abs_max,down_abs),
    }

def closeness_to_one(v):
    if pd.isna(v) or v<=0:return np.inf
    return abs(math.log(v))

def choose_cross(z):
    crosses=sustained_up_crosses(z)
    if not crosses:return None,[]
    geoms=[g for g in (geometry(z,i) for i in crosses) if g]
    if not geoms:return None,[]
    # Deterministic structural choice: most recent sustained cross.
    return geoms[-1],geoms

def render(z,case,g,out_dir):
    end=len(z); start=max(0,end-180)
    t=z.iloc[start:].copy().reset_index().rename(columns={"index":"orig_i"})
    factor=float(t.iloc[-1].ac/t.iloc[-1].close) if float(t.iloc[-1].close)>0 else 1.0
    t["price"]=t.ac/factor
    t["ma_raw"]=t.ma/factor
    x=np.arange(len(t))
    fig,ax=plt.subplots(figsize=(15,7))
    ax.plot(x,t.price,label="Close",linewidth=1.1)
    ax.plot(x,t.ma_raw,label=f'EMA{int(case["share_ma"])}',linewidth=1.5)

    cross_orig=g["cross_i"]
    hit=t.index[t.orig_i.eq(cross_orig)]
    if len(hit):
        cx=int(hit[-1]); ax.axvline(cx,linestyle="--",linewidth=1.0,label="sustained up-cross")
    ps=pd.Timestamp(g["pre_start_date"]); pe=pd.Timestamp(g["pre_end_date"]); ae=pd.Timestamp(g["post_end_date"])
    for a,b,label in [(ps,pe,"below-MA box"),(pd.Timestamp(g["cross_date"]),ae,"above-MA box")]:
        ids=t.index[(t.date>=a)&(t.date<=b)]
        if len(ids):ax.axvspan(int(ids.min()),int(ids.max()),alpha=.08,label=label)

    ax.set_title(f'{case["stock"]} | analysis {pd.Timestamp(case["analysis_date"]).date()} | EMA{int(case["share_ma"])} share 1:1 study')
    ax.grid(alpha=.15); ax.legend(loc="best")
    txt=(f'Vcur={g["ratio_vertical_current"]:.2f} Vmax={g["ratio_vertical_max"]:.2f} '
         f'Days={g["ratio_days"]:.2f} Area={g["ratio_area"]:.2f} '
         f'RectCur={g["ratio_rect_current"]:.2f} Intraday={g["ratio_intraday_extreme"]:.2f} '
         f'CrossMAcur={g["ratio_cross_ma_current"]:.2f}')
    fig.text(.01,.01,txt,fontsize=9)
    fig.tight_layout(rect=[0,.035,1,1])
    out_dir.mkdir(parents=True,exist_ok=True)
    p=out_dir/f'{case["stock"]}_{int(case["share_ma"])}_{pd.Timestamp(case["analysis_date"]).date()}.png'
    fig.savefig(p,dpi=150); plt.close(fig)
    return p

def summarize(df):
    metrics=[
        "ratio_vertical_current","ratio_vertical_max","ratio_days","ratio_area",
        "ratio_rect_current","ratio_rect_max","ratio_intraday_extreme",
        "ratio_cross_ma_current","ratio_cross_ma_max"
    ]
    rows=[]
    pos=df[df.label.eq("confirmed_1to1")]
    for m in metrics:
        vals=pd.to_numeric(pos[m],errors="coerce").dropna()
        if vals.empty:continue
        rows.append({
            "metric":m,"n":len(vals),
            "median":vals.median(),
            "mean_abs_log_error_to_1":np.mean(np.abs(np.log(vals[vals>0]))) if (vals>0).any() else np.nan,
            "within_20pct_of_1":float(((vals>=.8)&(vals<=1.2)).mean()),
            "within_35pct_of_1":float(((vals>=.65)&(vals<=1.35)).mean()),
            "min":vals.min(),"max":vals.max(),
        })
    return pd.DataFrame(rows).sort_values(["mean_abs_log_error_to_1","within_35pct_of_1"],ascending=[True,False])

def main():
    a=parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    panel=pd.concat([load(a.ko,"KO"),load(a.kq,"KQ")],ignore_index=True)
    cases=pd.read_csv(a.cases); cases["analysis_date"]=pd.to_datetime(cases.analysis_date)
    rows=[]; misses=[]
    for case in cases.to_dict("records"):
        q=panel[(panel.name.eq(str(case["stock"]).strip()))&(panel.date<=case["analysis_date"])].copy()
        if q.empty:
            misses.append({**case,"reason":"name_not_found"});continue
        sid=q.groupby("series_id").size().sort_values(ascending=False).index[0]
        z=feat(q[q.series_id.eq(sid)].sort_values("date"),int(case["share_ma"]))
        g,allg=choose_cross(z)
        if g is None:
            misses.append({**case,"reason":"no_sustained_up_cross"});continue
        rec={**case,**g,"series_id":sid,"market_date":z.iloc[-1].date,"num_sustained_crosses":len(allg)}
        rows.append(rec); render(z,case,g,a.out/"charts")
    out=pd.DataFrame(rows)
    out.to_csv(a.out/"share_1to1_geometry.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(misses).to_csv(a.out/"share_1to1_misses.csv",index=False,encoding="utf-8-sig")
    s=summarize(out) if not out.empty else pd.DataFrame()
    s.to_csv(a.out/"share_1to1_metric_summary.csv",index=False,encoding="utf-8-sig")
    print(f"cases={len(cases)} matched={len(out)} misses={len(misses)}")
    if not out.empty:
        cols=["analysis_date","stock","share_ma","label","cross_date","pre_days","post_days",
              "ratio_vertical_current","ratio_vertical_max","ratio_days","ratio_area",
              "ratio_rect_current","ratio_rect_max","ratio_intraday_extreme",
              "ratio_cross_ma_current","ratio_cross_ma_max"]
        print("\\n=== CASE GEOMETRY ===")
        print(out[cols].to_string(index=False))
        print("\\n=== METRIC SUMMARY (confirmed only) ===")
        print(s.to_string(index=False))

if __name__=="__main__":
    main()
