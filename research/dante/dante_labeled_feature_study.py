#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Feature study for dated official/public Dante examples.

Research-only:
- uses only data available on/before each published analysis date
- extracts transparent market features
- does NOT claim to reproduce proprietary indicators
"""
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd

EMAS=(5,15,20,33,56,60,112,224,448)

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--ko",type=Path,required=True)
    p.add_argument("--kq",type=Path,required=True)
    p.add_argument("--labels",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("dante_labeled_study"))
    return p.parse_args()

def load(path,exchange):
    x=pd.read_parquet(path).copy()
    x["date"]=pd.to_datetime(x["date"],errors="coerce")
    x["name"]=x.get("name","").astype(str)
    if "exchange" not in x:x["exchange"]=exchange
    if "adjusted_close" not in x:x["adjusted_close"]=x["close"]
    if "amount" not in x:x["amount"]=x["close"]*x["volume"]
    return x.sort_values(["series_id","date"])

def feat(g):
    z=g.sort_values("date").copy().reset_index(drop=True)
    factor=(z.adjusted_close/z.close).replace([np.inf,-np.inf],np.nan).ffill().bfill().fillna(1.0)
    z["ao"]=z.open*factor; z["ah"]=z.high*factor; z["al"]=z.low*factor; z["ac"]=z.adjusted_close
    for n in EMAS:
        z[f"ema{n}"]=z.ac.ewm(span=n,adjust=False,min_periods=n).mean()
    for n in (20,35,40):
        mid=z.ac.rolling(n,min_periods=n).mean()
        sd=z.ac.rolling(n,min_periods=n).std(ddof=0)
        z[f"bb{n}_upper2"]=mid+2*sd
    z["bb20_upper2_shift26"]=z["bb20_upper2"].shift(26)
    z["v5"]=z.volume.rolling(5,min_periods=5).mean()
    z["v20"]=z.volume.rolling(20,min_periods=20).mean()
    z["amount20"]=z.amount.rolling(20,min_periods=20).mean()
    prev=z.ac.shift(1)
    tr=pd.concat([(z.ah-z.al),(z.ah-prev).abs(),(z.al-prev).abs()],axis=1).max(axis=1)
    z["atr14"]=tr.rolling(14,min_periods=14).mean()
    z["ret1"]=z.ac.pct_change(fill_method=None)
    z["body_pct"]=(z.ac-z.ao)/z.ao
    rng=(z.ah-z.al).replace(0,np.nan)
    z["close_pos"]=(z.ac-z.al)/rng
    for n in (33,56,112,224,448):
        z[f"ema{n}_slope10"]=z[f"ema{n}"]/z[f"ema{n}"].shift(10)-1
        z[f"ema{n}_slope20"]=z[f"ema{n}"]/z[f"ema{n}"].shift(20)-1
    return z

def latest_up_cross(z,n,end_i):
    e=z[f"ema{n}"]
    cross=(z.ac>e)&(z.ac.shift(1)<=e.shift(1))
    idx=np.flatnonzero(cross.iloc[:end_i+1].fillna(False).to_numpy())
    return int(idx[-1]) if len(idx) else None

def share_candidates(z,n,i):
    out={}
    cross_i=latest_up_cross(z,n,i)
    out[f"ma{n}_last_up_cross_days"]=np.nan
    out[f"ma{n}_pre_down_amp_pct"]=np.nan
    out[f"ma{n}_current_up_amp_pct"]=np.nan
    out[f"ma{n}_mirror_ratio_candidate"]=np.nan
    out[f"ma{n}_below_run_before_cross"]=np.nan
    out[f"ma{n}_days_since_cross"]=np.nan
    if cross_i is None:return out
    out[f"ma{n}_last_up_cross_days"]=i-cross_i
    out[f"ma{n}_days_since_cross"]=i-cross_i
    j=cross_i-1; run=0
    while j>=0 and pd.notna(z.at[j,f"ema{n}"]) and float(z.at[j,"ac"])<float(z.at[j,f"ema{n}"]):
        run+=1; j-=1
    out[f"ma{n}_below_run_before_cross"]=run
    lo=max(0,cross_i-60)
    p=z.iloc[lo:cross_i]
    if len(p):
        denom=p[f"ema{n}"].replace(0,np.nan)
        down=((denom-p.ac)/denom).where(p.ac<denom)
        down_amp=float(down.max()) if down.notna().any() else np.nan
    else:down_amp=np.nan
    cur_ma=float(z.at[i,f"ema{n}"]) if pd.notna(z.at[i,f"ema{n}"]) else np.nan
    up_amp=float(z.at[i,"ac"]/cur_ma-1) if math.isfinite(cur_ma) and cur_ma>0 and z.at[i,"ac"]>cur_ma else np.nan
    out[f"ma{n}_pre_down_amp_pct"]=down_amp*100 if pd.notna(down_amp) else np.nan
    out[f"ma{n}_current_up_amp_pct"]=up_amp*100 if pd.notna(up_amp) else np.nan
    out[f"ma{n}_mirror_ratio_candidate"]=up_amp/down_amp if pd.notna(up_amp) and pd.notna(down_amp) and down_amp>0 else np.nan
    return out

def one_features(g,analysis_date):
    z=feat(g[g.date<=analysis_date].copy())
    if z.empty:return None
    i=len(z)-1; r=z.iloc[i]
    if pd.Timestamp(r.date).normalize()!=pd.Timestamp(analysis_date).normalize():
        # use last market session at or before analysis date
        pass
    rec={"market_date":r.date,"history_rows":len(z),"close":float(r.close),"amount20":float(r.amount20) if pd.notna(r.amount20) else np.nan}
    factor=float(r.ac/r.close) if float(r.close)>0 else 1.0
    for n in EMAS:
        rec[f"ema{n}"]=float(r[f"ema{n}"]/factor) if pd.notna(r[f"ema{n}"]) else np.nan
        rec[f"dist_ema{n}_pct"]=float((r.ac/r[f"ema{n}"]-1)*100) if pd.notna(r[f"ema{n}"]) and r[f"ema{n}"]>0 else np.nan
    for n in (33,56,112,224,448):
        rec[f"ema{n}_slope10_pct"]=float(r[f"ema{n}_slope10"]*100) if pd.notna(r[f"ema{n}_slope10"]) else np.nan
        rec[f"ema{n}_slope20_pct"]=float(r[f"ema{n}_slope20"]*100) if pd.notna(r[f"ema{n}_slope20"]) else np.nan
    rec["reverse_112_224_448"]=bool(pd.notna(r.ema448) and r.ema112<r.ema224<r.ema448)
    rec["above112"]=bool(pd.notna(r.ema112) and r.ac>=r.ema112)
    rec["above224"]=bool(pd.notna(r.ema224) and r.ac>=r.ema224)
    rec["ema5_above112"]=bool(pd.notna(r.ema112) and r.ema5>r.ema112)
    rec["ma112_224_gap_pct"]=float(abs(r.ema112-r.ema224)/r.ema224*100) if pd.notna(r.ema112) and pd.notna(r.ema224) and r.ema224>0 else np.nan
    vals=[r.ema112,r.ema224,r.ema448]
    rec["long_ma_compression_pct"]=float((max(vals)-min(vals))/r.ac*100) if all(pd.notna(v) for v in vals) and r.ac>0 else np.nan
    rec["volume_ratio20"]=float(r.volume/r.v20) if pd.notna(r.v20) and r.v20>0 else np.nan
    rec["volume_ratio5"]=float(r.volume/r.v5) if pd.notna(r.v5) and r.v5>0 else np.nan
    rec["body_pct"]=float(r.body_pct*100) if pd.notna(r.body_pct) else np.nan
    rec["close_pos"]=float(r.close_pos) if pd.notna(r.close_pos) else np.nan
    rec["atr14_pct"]=float(r.atr14/r.ac*100) if pd.notna(r.atr14) and r.ac>0 else np.nan
    for n in (20,35,40):
        u=r[f"bb{n}_upper2"]
        rec[f"dist_bb{n}_upper2_pct"]=float((r.ac/u-1)*100) if pd.notna(u) and u>0 else np.nan
    u=r["bb20_upper2_shift26"]
    rec["dist_bb20_upper2_shift26_pct"]=float((r.ac/u-1)*100) if pd.notna(u) and u>0 else np.nan
    for n in (20,60,120):
        w=z.tail(n)
        rec[f"drawdown_from_{n}d_high_pct"]=float((r.ac/w.ah.max()-1)*100) if len(w) else np.nan
        rec[f"rise_from_{n}d_low_pct"]=float((r.ac/w.al.min()-1)*100) if len(w) and w.al.min()>0 else np.nan
    rec["strong_up_candles_20d"]=int(((z.tail(20).ret1>=.05)&(z.tail(20).volume>=z.tail(20).v20*1.8)).sum())
    rec["max_volume_ratio20_last20"]=float((z.tail(20).volume/z.tail(20).v20).max()) if len(z)>=20 else np.nan
    for n in (60,112,224):rec.update(share_candidates(z,n,i))
    return rec

def summarize(df):
    rows=[]
    groups=[
        ("watermelon",df.has_watermelon),
        ("blue_dot",df.has_blue_dot),
        ("rainbow",df.has_rainbow),
        ("stop_hit",df.stop_hit),
    ]
    features=[
        "dist_ema112_pct","dist_ema224_pct","ma112_224_gap_pct","long_ma_compression_pct",
        "ema112_slope20_pct","ema224_slope20_pct","volume_ratio20","atr14_pct",
        "dist_bb20_upper2_pct","dist_bb35_upper2_pct","dist_bb40_upper2_pct","dist_bb20_upper2_shift26_pct",
        "ma60_mirror_ratio_candidate","ma112_mirror_ratio_candidate","ma224_mirror_ratio_candidate",
    ]
    for name,mask in groups:
        for flag in (False,True):
            q=df[mask.eq(flag)]
            if q.empty:continue
            rec={"group":name,"value":flag,"n":len(q)}
            for f in features:
                rec[f"{f}_median"]=pd.to_numeric(q[f],errors="coerce").median()
            rows.append(rec)
    return pd.DataFrame(rows)

def main():
    a=parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    panel=pd.concat([load(a.ko,"KO"),load(a.kq,"KQ")],ignore_index=True)
    labels=pd.read_csv(a.labels)
    labels["analysis_date"]=pd.to_datetime(labels.analysis_date)
    rows=[]; misses=[]
    for lab in labels.to_dict("records"):
        name=str(lab["stock"]).strip()
        q=panel[(panel.name.astype(str).str.strip()==name)&(panel.date<=lab["analysis_date"])].copy()
        if q.empty:
            misses.append({**lab,"reason":"name_not_found"}); continue
        sid=q.groupby("series_id").date.max().idxmax()
        g=panel[panel.series_id.eq(sid)&(panel.date<=lab["analysis_date"])].sort_values("date")
        market_date=pd.Timestamp(g.iloc[-1].date)
        if (pd.Timestamp(lab["analysis_date"])-market_date).days>7:
            misses.append({**lab,"reason":"stale_history","market_date":market_date}); continue
        f=one_features(g,pd.Timestamp(lab["analysis_date"]))
        if not f:
            misses.append({**lab,"reason":"feature_failed"}); continue
        vip=str(lab.get("vip_indicators",""))
        rec={**lab,**f,
             "series_id":sid,
             "has_watermelon":"수박" in vip,
             "has_blue_dot":"파란점선" in vip,
             "has_rainbow":"레인보우" in vip,
             "has_pink_arrow":"분홍" in vip,
             "has_green_arrow":"초록" in vip}
        rows.append(rec)
    out=pd.DataFrame(rows)
    out.to_csv(a.out/"official_example_features.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(misses).to_csv(a.out/"official_example_misses.csv",index=False,encoding="utf-8-sig")
    if not out.empty:
        summarize(out).to_csv(a.out/"official_example_feature_summary.csv",index=False,encoding="utf-8-sig")
        print(f"labels={len(labels)} matched={len(out)} misses={len(misses)}")
        print("\nSignal counts:")
        for c in ["has_watermelon","has_blue_dot","has_rainbow","has_pink_arrow","has_green_arrow","stop_hit"]:
            print(c, out[c].value_counts(dropna=False).to_dict())
        print("\nSelected features:")
        cols=["analysis_date","stock","outcome","has_watermelon","has_blue_dot","has_rainbow",
              "reverse_112_224_448","dist_ema112_pct","dist_ema224_pct","ma112_224_gap_pct",
              "ema112_slope20_pct","ema224_slope20_pct","volume_ratio20",
              "dist_bb20_upper2_pct","dist_bb35_upper2_pct","dist_bb40_upper2_pct",
              "dist_bb20_upper2_shift26_pct","ma60_mirror_ratio_candidate",
              "ma112_mirror_ratio_candidate","ma224_mirror_ratio_candidate"]
        print(out[cols].to_string(index=False))

if __name__=="__main__":
    main()
