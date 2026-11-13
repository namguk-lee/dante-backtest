#!/usr/bin/env python3
"""Experimental Dante Bowl-2 / concrete base scanner; never changes production.
Requires dante_classic_scanner.py. All historical features stop at the snapshot.
These are research proxies, NOT proprietary Dante formulas.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def find_bowl_v3(z, min_base=60):
    n = len(z)
    if n < 520:
        return None
    approach_i = next((i for i in range(max(448, n-60), n)
                       if pd.notna(z.at[i, "ema224"]) and
                       -0.10 <= float(z.at[i, "ac"]/z.at[i, "ema224"]-1) <= 0.10), None)
    if approach_i is None:
        return None
    end = approach_i-5
    left = max(448, approach_i-300)
    if end-left < min_base+10:
        return None
    peak_end = approach_i-35
    if peak_end <= left:
        return None
    peak_i = int(z.loc[left:peak_end, "ah"].idxmax())
    peak = float(z.at[peak_i, "ah"])
    if peak <= 0 or not np.isfinite(peak):
        return None
    closes = z.ac.to_numpy(dtype=float)
    lows = z.al.to_numpy(dtype=float)
    for start in range(max(peak_i+10, end-260), end-min_base+2):
        v = closes[start:end+1]
        if len(v) < min_base or not np.isfinite(v).all() or (v<=0).any():
            continue
        first = v[:min(20, len(v)//3)]
        if float(np.median(first)) > peak*0.80:
            continue
        q10,q90 = np.quantile(v,[0.1,0.9])
        cr = float(q90/q10-1) if q10>0 else np.inf
        cn = float(v[-1]/v[0]-1)
        maxdd = float(np.min(v/np.maximum.accumulate(v)-1))
        fresh_low = float(np.min(v[len(first):])) < float(np.min(first))*0.92
        slope = float(np.polyfit(np.arange(len(v)),v/float(np.median(v)),1)[0]*(len(v)-1))
        if not(cr<=.30 and abs(cn)<=.20 and maxdd>=-.25 and not fresh_low and slope>=-.12):
            continue
        decline = float(np.min(lows[peak_i:start+1])/peak-1)
        if decline > -.20 or end-start+1 < start-peak_i:
            continue
        break
    else:
        return None
    bottom_i = start+int(np.argmin(lows[start:end+1]))
    base = closes[start:approach_i+1]
    q10,q90 = np.quantile(base,[.1,.9])
    below = 0
    for j in range(approach_i-1,-1,-1):
        if pd.isna(z.at[j,"ema224"]) or not float(z.at[j,"ac"]) < float(z.at[j,"ema224"]):
            break
        below += 1
    return dict(approach_i=approach_i,peak_i=peak_i,base_start_i=start,bottom_i=bottom_i,
                core_end_i=end,below224_days_before_approach=below,
                bowl1_days=start-peak_i,bowl2_days=approach_i-start,
                bowl2_core_days=end-start+1,
                bowl2_over_bowl1=(approach_i-start)/(start-peak_i),
                bowl1_decline_pct=decline,
                bowl2_range_pct=q90/q10-1 if q10>0 else np.nan,
                bowl2_net_change_pct=float(base[-1]/base[0]-1),
                bowl2_core_range_pct=cr,bowl2_core_net_change_pct=cn)


def eligible(r):
    return bool(r.get("four_month_below224",False)) and -5<=float(r["dist224_pct"])<=8 and r["state"] in ("BOWL3_PRE224","BOWL3_224_HOLD")


def scan_snapshot(market, date, classic, mode, minimum):
    past=market[market.date<=date]
    active=set(past.loc[past.date.eq(date),"series_id"].astype(str))
    old=classic.find_bowl
    if mode=="v3": classic.find_bowl=find_bowl_v3
    try:
        found=[]
        for _,g in past[past.series_id.astype(str).isin(active)].groupby("series_id",sort=False):
            if len(g)<520: continue
            rec,_=classic.classify_one(g, minimum)
            if rec and eligible(rec): found.append(rec)
        return found
    finally:
        classic.find_bowl=old


def summary(df,group):
    v=df[df["group"]==group] if len(df) else df
    result={"group":group,"signals":len(v),"unique_stocks":v.code.nunique() if len(v) else 0}
    for h in (20,60):
        x=v[f"forward_{h}d_pct"].dropna().astype(float) if len(v) else pd.Series(dtype=float)
        result.update({f"n_{h}d":len(x),f"mean_{h}d":x.mean() if len(x) else np.nan,
                       f"median_{h}d":x.median() if len(x) else np.nan,
                       f"win_{h}d":100*(x>0).mean() if len(x) else np.nan})
    return result


def check_synthetic():
    import pandas as pd
    def zcase(kind):
        n=650
        p=np.full(n,130.,dtype=float)
        p[450:501]=np.linspace(150,70,51)
        p[501:n-5]=70+np.sin(np.arange(n-5-501)/6)*2
        if kind=="short":
            p[501:n-30]=np.linspace(70,30,n-30-501)
            p[n-30:n-5]=70+np.sin(np.arange(25)/6)*2
        if kind=="newlow": p[565:575]=50
        if kind=="drift": p[501:n-5]=np.linspace(76,55,n-5-501)
        p[n-5:]=[85,88,93,97,99]
        return pd.DataFrame({"ac":p,"ah":p*1.015,"al":p*.985,"ema224":np.full(n,100.)})
    checks={"stable":True,"short":False,"newlow":False,"drift":False}
    for case,expected in checks.items():
        got=bool(find_bowl_v3(zcase(case)))
        print("TEST",case,got,"expected",expected,flush=True)
        assert got==expected,case


def main():
    import dante_classic_scanner as classic
    p=argparse.ArgumentParser()
    p.add_argument("--ko",type=Path,required=True)
    p.add_argument("--kq",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--historical",action="store_true")
    p.add_argument("--min-turnover",type=float,default=5000000000)
    a=p.parse_args()
    check_synthetic()
    a.out.mkdir(parents=True,exist_ok=True)
    market=pd.concat([classic.load(a.ko,"KO"),classic.load(a.kq,"KQ")],ignore_index=True)
    dates=sorted(pd.to_datetime(market.date.unique()))
    months=["2025-06","2025-09","2025-12","2026-03","2026-06","2026-08"] if a.historical else []
    snapshots=[max(d for d in dates if d.strftime("%Y-%m")==m)
               for m in months if any(d.strftime("%Y-%m")==m for d in dates)]
    latest=dates[-1]
    if latest not in snapshots: snapshots.append(latest)
    by_series={str(sid):g.sort_values("date") for sid,g in market.groupby("series_id",sort=False)}
    rows=[]
    for asof in snapshots:
        print("AS_OF",asof.date(),flush=True)
        for mode in ("v2","v3"):
            results=scan_snapshot(market,asof,classic,mode,a.min_turnover)
            print(mode,"eligible",len(results),flush=True)
            for r in results:
                row={k:v for k,v in r.items() if not k.startswith("_idx_")}
                row.update(snapshot=str(asof.date()),group=mode,base_days=int(r["bowl2_core_days"]))
                g=by_series[str(r["series_id"])]
                f=g[g.date>asof]
                entry=g[g.date==asof].adjusted_close.iloc[-1]
                for h in (20,60):
                    row[f"forward_{h}d_pct"]=(100*(f.adjusted_close.iloc[h-1]/entry-1)-.40) if len(f)>=h else np.nan
                rows.append(row)
    df=pd.DataFrame(rows)
    if len(df)==0: df=pd.DataFrame(columns=["snapshot","group","code","name","base_days","forward_20d_pct","forward_60d_pct"])
    df.to_csv(a.out/"candidates.csv",index=False,encoding="utf-8-sig")
    hist=df[df.snapshot!=str(latest.date())]
    current=df[df.snapshot==str(latest.date())]
    summaries=[]
    for label,part in (("historical",hist),("latest",current)):
        for group in ("v2","v3"):
            sub=part[part.group==group].copy()
            s=summary(sub,group)
            s["period"]=label
            summaries.append(s)
            if group=="v3":
                s90=summary(sub[sub.base_days>=90],"v3") if False else None
                s90=summary(sub.assign(group="v3_90")[sub.base_days>=90],"v3_90")
                s90["period"]=label
                summaries.append(s90)
    pd.DataFrame(summaries).to_csv(a.out/"summary.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame({"snapshot":[str(x.date()) for x in snapshots]}).to_csv(a.out/"snapshots.csv",index=False)
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)

if __name__=="__main__": main()
