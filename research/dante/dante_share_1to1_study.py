#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reverse-engineer candidate geometries for Dante's public 'share 1:1' examples.

Research only. No proprietary formula is claimed.
Each example is evaluated only with data available on/before its analysis_date.

V2 adds candidates motivated by public lecture wording:
- candles above a reference MA = buy-side share
- candles below it = sell-side share
- 50:50 / "반반" is described as a transition
- price may need to recover roughly as far above the MA as the prior sell-side
  excursion below it

Accordingly this study compares:
1) candle-count share ratios over fixed windows
2) normalized distance/area ratios over the same windows
3) structural-cross swing-height candidates
"""
from __future__ import annotations
import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import FinanceDataReader as fdr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COUNT_WINDOWS=(20,40,60,80,112,120,160,224,252,336,448)

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
    x["code"]=x["code"].astype(str).str.replace(r"\.0$","",regex=True).str.zfill(6)
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

def safe_ratio(a,b):
    if pd.isna(a) or pd.isna(b) or b==0:return np.nan
    return float(a/b)

def fetch_fallback(code,name,analysis_date):
    """Fallback only for labeled research cases missing from the local panel."""
    try:
        end=(pd.Timestamp(analysis_date)+pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        q=fdr.DataReader("NAVER:"+str(code).zfill(6),"2022-01-01",end)
        if q is None or q.empty:return pd.DataFrame()
        q=q.reset_index(); q.columns=[str(x).lower() for x in q.columns]
        q["date"]=pd.to_datetime(q["date"],errors="coerce")
        for col in ("open","high","low","close","volume"):
            q[col]=pd.to_numeric(q[col],errors="coerce")
        q=q.dropna(subset=["date","open","high","low","close","volume"])
        q["adjusted_close"]=q["close"]
        q["code"]=str(code).zfill(6); q["name"]=name
        q["exchange"]="FALLBACK"; q["series_id"]="FALLBACK:"+str(code).zfill(6)
        return q[q.date<=pd.Timestamp(analysis_date)].sort_values("date")
    except Exception:
        return pd.DataFrame()

def prior_down_cross(z,cross_i,ma_n):
    """Last meaningful transition from above-MA to below-MA before the recovery cross."""
    start=max(1,cross_i-max(120,ma_n*3))
    cand=[]
    for i in range(start,cross_i):
        if pd.isna(z.at[i,"ma"]) or pd.isna(z.at[i-1,"ma"]):continue
        if z.at[i,"ac"]<z.at[i,"ma"] and z.at[i-1,"ac"]>=z.at[i-1,"ma"]:
            nxt=z.iloc[i:min(i+7,cross_i)]
            if len(nxt) and float((nxt.ac<nxt.ma).mean())>=.60:
                cand.append(i)
    return cand[-1] if cand else None

def cycle_features(z,cross_i,ma_n):
    """Measure the structurally selected sell->buy cycle around the reference MA."""
    dc=prior_down_cross(z,cross_i,ma_n)
    if dc is None:
        start=max(0,cross_i-max(60,ma_n))
    else:
        start=dc
    seg=z.iloc[start:].copy()
    seg=seg[pd.notna(seg.ma)]
    pre=z.iloc[start:cross_i].copy()
    post=z.iloc[cross_i:].copy()
    pre=pre[pd.notna(pre.ma)]; post=post[pd.notna(post.ma)]
    out={
        "cycle_start_i":start,
        "cycle_start_date":z.at[start,"date"] if start<len(z) else pd.NaT,
        "cycle_down_cross_date":z.at[dc,"date"] if dc is not None else pd.NaT,
        "cycle_days":len(seg),
        "cycle_pre_days":len(pre),
        "cycle_post_days":len(post),
    }
    if seg.empty:return out
    above=int((seg.ac>=seg.ma).sum()); below=int((seg.ac<seg.ma).sum())
    out["cycle_count_ratio"]=safe_ratio(above,below)
    out["cycle_above_share"]=safe_ratio(above,len(seg))
    up_area=float(seg.dist.clip(lower=0).sum()); dn_area=float((-seg.dist).clip(lower=0).sum())
    out["cycle_area_ratio"]=safe_ratio(up_area,dn_area)
    up_mean=float(seg.loc[seg.dist>0,"dist"].mean()) if (seg.dist>0).any() else np.nan
    dn_mean=float((-seg.loc[seg.dist<0,"dist"]).mean()) if (seg.dist<0).any() else np.nan
    out["cycle_mean_dist_ratio"]=safe_ratio(up_mean,dn_mean)
    neg_depth=float((-pre.dist).clip(lower=0).max()) if not pre.empty else np.nan
    pos_current=float(max(0,z.iloc[-1].dist))
    pos_max=float(post.dist.clip(lower=0).max()) if not post.empty else np.nan
    out["cycle_height_current_ratio"]=safe_ratio(pos_current,neg_depth)
    out["cycle_height_max_ratio"]=safe_ratio(pos_max,neg_depth)
    out["cycle_duration_ratio"]=safe_ratio(len(post),len(pre))
    if not pre.empty:
        low_i=int(pre.al.idxmin())
        out["cycle_low_date"]=z.at[low_i,"date"]
        out["cycle_low_to_analysis_days"]=len(z)-1-low_i
        out["cycle_downcross_to_low_days"]=low_i-start
        out["cycle_time_symmetry_ratio"]=safe_ratio(len(z)-1-low_i,low_i-start)
    else:
        out["cycle_low_date"]=pd.NaT
        out["cycle_low_to_analysis_days"]=np.nan
        out["cycle_downcross_to_low_days"]=np.nan
        out["cycle_time_symmetry_ratio"]=np.nan
    return out

def structural_up_crosses(z,ma_n,lookback=None):
    """Ignore one-day whipsaws: require a meaningful sell-side share before cross."""
    if lookback is None:
        lookback=min(len(z),max(180,ma_n*2))
    start=max(1,len(z)-lookback)
    pre_w=max(20,min(80,ma_n//4))
    out=[]
    for i in range(start,len(z)-2):
        if pd.isna(z.at[i,"ma"]) or pd.isna(z.at[i-1,"ma"]):continue
        if not(z.at[i,"ac"]>=z.at[i,"ma"] and z.at[i-1,"ac"]<z.at[i-1,"ma"]):continue
        pre=z.iloc[max(0,i-pre_w):i]
        if len(pre)<max(10,pre_w//2):continue
        below_share=float((pre.ac<pre.ma).mean())
        nxt=z.iloc[i:min(i+7,len(z))]
        post_share=float((nxt.ac>=nxt.ma).mean()) if len(nxt) else 0.0
        if below_share>=.60 and post_share>=.60:
            out.append((i,below_share,post_share))
    return out

def contiguous_runs(z,cross_i):
    j=cross_i-1
    while j>=0 and pd.notna(z.at[j,"ma"]) and z.at[j,"ac"]<z.at[j,"ma"]:
        j-=1
    pre_start=j+1; pre_end=cross_i-1
    k=cross_i
    while k+1<len(z) and pd.notna(z.at[k+1,"ma"]) and z.at[k+1,"ac"]>=z.at[k+1,"ma"]:
        k+=1
    return pre_start,pre_end,cross_i,k

def latest_local_trough_features(z,cross_i):
    """Compare the current buy-side wave with the latest meaningful sell-side trough.

    This is closer to the public visual explanation than fixed windows:
    locate local minima in distance-to-MA before the structural up-cross, then
    compare the current distance above the MA with that sell-side excursion.
    Both percent-normalized and absolute-price box heights are tested.
    """
    rec={}
    current_close=float(z.iloc[-1].ac)
    current_ma=float(z.iloc[-1].ma) if pd.notna(z.iloc[-1].ma) else np.nan
    current_pct=(current_close-current_ma)/current_ma if pd.notna(current_ma) and current_ma>0 else np.nan
    current_abs=current_close-current_ma if pd.notna(current_ma) else np.nan

    for lookback in (15,20,30,40,60,80):
        s=max(3,cross_i-lookback); e=cross_i
        mins=[]
        for i in range(s,e):
            if i<3 or i+3>=len(z) or pd.isna(z.at[i,"ma"]):continue
            d=float(z.at[i,"dist"])
            if d>=0:continue
            if d<=float(z.loc[i-3:i+3,"dist"].min())+1e-12:
                mins.append(i)
        if not mins:
            q=z.iloc[s:e]
            q=q[q.dist<0]
            if not q.empty:mins=[int(q.dist.idxmin())]
        if not mins:
            rec[f"local_trough_date_{lookback}"]=pd.NaT
            rec[f"ratio_local_pct_{lookback}"]=np.nan
            rec[f"ratio_local_abs_{lookback}"]=np.nan
            rec[f"ratio_local_low_abs_{lookback}"]=np.nan
            continue

        # Latest local sell-side trough is the wave immediately preceding the recovery.
        ti=mins[-1]
        trough_close=float(z.at[ti,"ac"])
        trough_low=float(z.at[ti,"al"])
        trough_ma=float(z.at[ti,"ma"])
        sell_pct=max(0,(trough_ma-trough_close)/trough_ma) if trough_ma>0 else np.nan
        sell_abs=max(0,trough_ma-trough_close)
        sell_low_abs=max(0,trough_ma-trough_low)
        rec[f"local_trough_date_{lookback}"]=z.at[ti,"date"]
        rec[f"local_trough_dist_pct_{lookback}"]=sell_pct*100 if pd.notna(sell_pct) else np.nan
        rec[f"ratio_local_pct_{lookback}"]=safe_ratio(max(0,current_pct),sell_pct)
        rec[f"ratio_local_abs_{lookback}"]=safe_ratio(max(0,current_abs),sell_abs)
        rec[f"ratio_local_low_abs_{lookback}"]=safe_ratio(max(0,current_abs),sell_low_abs)

    return rec

def window_share_features(z):
    rec={}
    valid=z[pd.notna(z.ma)].copy()
    if valid.empty:return rec
    for w in COUNT_WINDOWS:
        q=valid.tail(w)
        if len(q)<max(10,min(w,20)):continue
        above=int((q.ac>=q.ma).sum()); below=int((q.ac<q.ma).sum())
        rec[f"count_ratio_{w}"]=safe_ratio(above,below)
        rec[f"above_share_{w}"]=safe_ratio(above,len(q))
        up=float(q.dist.clip(lower=0).sum())
        dn=float((-q.dist).clip(lower=0).sum())
        rec[f"area_ratio_{w}"]=safe_ratio(up,dn)
        rec[f"mean_dist_ratio_{w}"]=safe_ratio(
            q.loc[q.dist>0,"dist"].mean() if (q.dist>0).any() else np.nan,
            (-q.loc[q.dist<0,"dist"]).mean() if (q.dist<0).any() else np.nan
        )
    above=int((valid.ac>=valid.ma).sum()); below=int((valid.ac<valid.ma).sum())
    rec["count_ratio_full_valid"]=safe_ratio(above,below)
    rec["above_share_full_valid"]=safe_ratio(above,len(valid))
    rec["area_ratio_full_valid"]=safe_ratio(
        float(valid.dist.clip(lower=0).sum()),
        float((-valid.dist).clip(lower=0).sum())
    )
    return rec

def geometry(z,cross_i,ma_n):
    ps,pe,as_,ae=contiguous_runs(z,cross_i)
    if pe<ps or ae<as_:return None
    pre=z.iloc[ps:pe+1].copy()
    post=z.iloc[as_:ae+1].copy()
    if pre.empty or post.empty:return None

    current=float(z.iloc[-1].ac)
    ma0=float(z.at[cross_i,"ma"])
    post_height_current=float(max(0,z.iloc[-1].dist))
    post_height_max=float(post.dist.clip(lower=0).max())

    pre_depth_contig=float((-pre.dist).clip(lower=0).max())
    post_high_contig=float(((post.ah-post.ma)/post.ma).clip(lower=0).max())
    pre_low_contig=float(((pre.ma-pre.al)/pre.ma).clip(lower=0).max())

    # Broader sell-side boxes before the structural cross, tied to MA period.
    broad={}
    for mult,label in ((.5,"halfma"),(1.0,"1ma"),(2.0,"2ma")):
        w=max(20,int(round(ma_n*mult)))
        q=z.iloc[max(0,cross_i-w):cross_i].copy()
        q=q[pd.notna(q.ma)]
        if q.empty:
            broad[f"ratio_broad_depth_{label}_current"]=np.nan
            broad[f"ratio_broad_depth_{label}_max"]=np.nan
            broad[f"ratio_broad_intraday_{label}_current"]=np.nan
            continue
        depth=float((-q.dist).clip(lower=0).max())
        low_ext=float(((q.ma-q.al)/q.ma).clip(lower=0).max())
        broad[f"ratio_broad_depth_{label}_current"]=safe_ratio(post_height_current,depth)
        broad[f"ratio_broad_depth_{label}_max"]=safe_ratio(post_height_max,depth)
        broad[f"ratio_broad_intraday_{label}_current"]=safe_ratio(post_height_current,low_ext)

    ma_cross_down=(ma0-float(pre.al.min()))/ma0 if ma0>0 else np.nan
    ma_cross_up_current=(current-ma0)/ma0 if ma0>0 else np.nan

    out={
        "cross_i":cross_i,
        "cross_date":z.at[cross_i,"date"],
        "pre_start_date":z.at[ps,"date"],
        "pre_end_date":z.at[pe,"date"],
        "post_end_date":z.at[ae,"date"],
        "pre_days":len(pre),
        "post_days":len(post),
        "ratio_contig_days":safe_ratio(len(post),len(pre)),
        "ratio_contig_vertical_current":safe_ratio(post_height_current,pre_depth_contig),
        "ratio_contig_vertical_max":safe_ratio(post_height_max,pre_depth_contig),
        "ratio_contig_intraday":safe_ratio(post_high_contig,pre_low_contig),
        "ratio_cross_ma_current":safe_ratio(ma_cross_up_current,ma_cross_down),
    }
    out.update(broad)
    out.update(latest_local_trough_features(z,cross_i))
    out.update(window_share_features(z))
    out.update(cycle_features(z,cross_i,ma_n))
    return out

def choose_cross(z,ma_n):
    crosses=structural_up_crosses(z,ma_n)
    if not crosses:return None,[]
    geoms=[]
    for i,bshare,ashare in crosses:
        g=geometry(z,i,ma_n)
        if g:
            g["pre_below_share_at_cross"]=bshare
            g["post_above_share_at_cross"]=ashare
            geoms.append(g)
    if not geoms:return None,[]
    # Most recent *structural* cross, not the latest one-day recross.
    return geoms[-1],geoms

def render(z,case,g,out_dir):
    end=len(z); start=max(0,end-260)
    t=z.iloc[start:].copy().reset_index().rename(columns={"index":"orig_i"})
    factor=float(t.iloc[-1].ac/t.iloc[-1].close) if float(t.iloc[-1].close)>0 else 1.0
    t["price"]=t.ac/factor; t["ma_raw"]=t.ma/factor
    x=np.arange(len(t))
    fig,ax=plt.subplots(figsize=(15,7))
    ax.plot(x,t.price,label="Close",linewidth=1.1)
    ax.plot(x,t.ma_raw,label=f'EMA{int(case["share_ma"])}',linewidth=1.5)
    hit=t.index[t.orig_i.eq(int(g["cross_i"]))]
    if len(hit):ax.axvline(int(hit[-1]),linestyle="--",linewidth=1.0,label="structural up-cross")
    ax.set_title(f'{case["stock"]} | analysis {pd.Timestamp(case["analysis_date"]).date()} | EMA{int(case["share_ma"])} share study')
    ax.grid(alpha=.15); ax.legend(loc="best")
    txt=(
        f'count(1MA)={g.get(f"count_ratio_{int(case["share_ma"])}",np.nan):.2f} '
        f'count(2MA)={g.get(f"count_ratio_{int(case["share_ma"])*2}",np.nan):.2f} '
        f'broadHeight(1MA)={g.get("ratio_broad_depth_1ma_current",np.nan):.2f} '
        f'broadHeight(2MA)={g.get("ratio_broad_depth_2ma_current",np.nan):.2f}'
    )
    fig.text(.01,.01,txt,fontsize=9)
    fig.tight_layout(rect=[0,.035,1,1])
    out_dir.mkdir(parents=True,exist_ok=True)
    p=out_dir/f'{case["stock"]}_{int(case["share_ma"])}_{pd.Timestamp(case["analysis_date"]).date()}_v2.png'
    fig.savefig(p,dpi=150); plt.close(fig)
    return p

def metric_summary(df):
    pos=df[df.label.eq("confirmed_1to1")]
    metric_cols=[c for c in df.columns if c.startswith("count_ratio_") or c.startswith("area_ratio_") or c.startswith("mean_dist_ratio_") or c.startswith("ratio_")]
    rows=[]
    for m in metric_cols:
        vals=pd.to_numeric(pos[m],errors="coerce").dropna()
        vals=vals[vals>0]
        if len(vals)<2:continue
        rows.append({
            "metric":m,"n":len(vals),"median":float(vals.median()),
            "mean_abs_log_error_to_1":float(np.mean(np.abs(np.log(vals)))),
            "within_20pct_of_1":float(((vals>=.8)&(vals<=1.2)).mean()),
            "within_35pct_of_1":float(((vals>=.65)&(vals<=1.35)).mean()),
            "at_or_above_1":float((vals>=1).mean()),
            "min":float(vals.min()),"max":float(vals.max())
        })
    return pd.DataFrame(rows).sort_values(
        ["within_35pct_of_1","mean_abs_log_error_to_1"],
        ascending=[False,True]
    )

def main():
    a=parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    panel=pd.concat([load(a.ko,"KO"),load(a.kq,"KQ")],ignore_index=True)
    cases=pd.read_csv(a.cases,dtype={"code":str})
    cases["code"]=cases.code.astype(str).str.zfill(6)
    cases["analysis_date"]=pd.to_datetime(cases.analysis_date)
    rows=[]; misses=[]
    for case in cases.to_dict("records"):
        q=panel[(panel.code.eq(case["code"]))&(panel.date<=case["analysis_date"])].copy()
        if q.empty:
            q=panel[(panel.name.eq(str(case["stock"]).strip()))&(panel.date<=case["analysis_date"])].copy()
        used_fallback=False
        if q.empty:
            q=fetch_fallback(case["code"],case["stock"],case["analysis_date"])
            used_fallback=not q.empty
        if q.empty:
            misses.append({**case,"reason":"stock_not_found"});continue
        sid=q.groupby("series_id").size().sort_values(ascending=False).index[0]
        z=feat(q[q.series_id.eq(sid)].sort_values("date"),int(case["share_ma"]))
        g,allg=choose_cross(z,int(case["share_ma"]))
        if g is None:
            # Count/share metrics do not require a structural cross, so retain case.
            g=window_share_features(z)
            g.update({"cross_i":np.nan,"cross_date":pd.NaT,"pre_start_date":pd.NaT,"pre_end_date":pd.NaT,"post_end_date":pd.NaT})
        rec={**case,**g,"series_id":sid,"market_date":z.iloc[-1].date,"num_structural_crosses":len(allg),"used_fallback":used_fallback}
        rows.append(rec)
        if pd.notna(rec.get("cross_i")):render(z,case,rec,a.out/"charts")
    out=pd.DataFrame(rows)
    out.to_csv(a.out/"share_1to1_geometry_v3.csv",index=False,encoding="utf-8-sig")
    all_cross_rows=[]
    for case in cases.to_dict("records"):
        q=panel[(panel.code.eq(case["code"]))&(panel.date<=case["analysis_date"])].copy()
        if q.empty:
            q=fetch_fallback(case["code"],case["stock"],case["analysis_date"])
        if q.empty:continue
        sid=q.groupby("series_id").size().sort_values(ascending=False).index[0]
        z=feat(q[q.series_id.eq(sid)].sort_values("date"),int(case["share_ma"]))
        for rank,(cross_i,bshare,ashare) in enumerate(structural_up_crosses(z,int(case["share_ma"])),1):
            gg=geometry(z,cross_i,int(case["share_ma"]))
            if gg:
                all_cross_rows.append({**case,**gg,"candidate_rank":rank,"pre_below_share_at_cross":bshare,"post_above_share_at_cross":ashare})
    pd.DataFrame(all_cross_rows).to_csv(a.out/"share_1to1_all_structural_crosses_v3.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(misses).to_csv(a.out/"share_1to1_misses_v2.csv",index=False,encoding="utf-8-sig")
    s=metric_summary(out) if not out.empty else pd.DataFrame()
    s.to_csv(a.out/"share_1to1_metric_summary_v3.csv",index=False,encoding="utf-8-sig")
    print(f"cases={len(cases)} matched={len(out)} misses={len(misses)}")
    print("\\n=== CASES ===")
    show=["analysis_date","stock","code","share_ma","label","market_date","cross_date","num_structural_crosses"]
    for w in (60,112,120,224,252,448):
        if f"count_ratio_{w}" in out.columns:show.append(f"count_ratio_{w}")
    for m in ("cycle_count_ratio","cycle_area_ratio","cycle_mean_dist_ratio","cycle_height_current_ratio","cycle_height_max_ratio","cycle_duration_ratio","cycle_time_symmetry_ratio","count_ratio_full_valid","ratio_broad_depth_halfma_current","ratio_broad_depth_1ma_current","ratio_broad_depth_2ma_current","ratio_cross_ma_current"):
        if m in out.columns:show.append(m)
    print(out[show].to_string(index=False))
    print("\\n=== BEST METRICS (confirmed only) ===")
    print(s.head(30).to_string(index=False))

if __name__=="__main__":
    main()
