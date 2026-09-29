#!/usr/bin/env python3
"""Locked V4 filter validation on V3.1 signal/trade outputs.

The rule is intentionally simple and fixed. TEST was already inspected during
V3.1 analysis, so TEST is labeled post-hoc rather than pristine OOS.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

RULE = {
    "entry_gap_max": 0.05,
    "convergence_gap_max": 0.05,
    "headroom448_min": 0.05,
    "below80_min": 80,
    "volume_ratio_max": 5.0,
}

def summarize(q, value="fwd60"):
    if q.empty:
        return {"n":0,"mean":np.nan,"median":np.nan,"trim10_mean":np.nan,"win_rate":np.nan}
    x=pd.to_numeric(q[value],errors="coerce").dropna().sort_values().to_numpy()
    k=int(len(x)*.10)
    trim=x[k:len(x)-k] if len(x)>=10 and 2*k<len(x) else x
    return {"n":len(x),"mean":float(np.mean(x)),"median":float(np.median(x)),
            "trim10_mean":float(np.mean(trim)),"win_rate":float(np.mean(x>0))}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--signals",type=Path,required=True)
    ap.add_argument("--trades",type=Path,required=True)
    ap.add_argument("--out",type=Path,default=Path("dante_results"))
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)

    s=pd.read_csv(a.signals); t=pd.read_csv(a.trades)
    s["signal_date"]=pd.to_datetime(s.signal_date); t["signal_date"]=pd.to_datetime(t.signal_date)
    s["year"]=s.signal_date.dt.year
    s["entry_gap"]=s.entry/s.ema224-1
    s["convergence_gap"]=(s.ema224-s.ema112)/s.ema224
    s["headroom448"]=s.ema448/s.entry-1

    mask=(s.entry_gap<=RULE["entry_gap_max"]) & \
         (s.convergence_gap<=RULE["convergence_gap_max"]) & \
         (s.headroom448>=RULE["headroom448_min"]) & \
         (s.below_80>=RULE["below80_min"]) & \
         (s.volume_ratio<=RULE["volume_ratio_max"])
    c=s[mask].copy()
    c.to_csv(a.out/"v4_candidate_signals.csv",index=False,encoding="utf-8-sig")

    rows=[]
    for label,q in [("BASELINE",s),("V4_LOCKED",c)]:
        for sp,z in q.groupby("split"):
            r={"rule":label,"split":sp,**summarize(z)}
            r["evidence_status"]="POSTHOC_TEST" if sp=="TEST" else "DEVELOPMENT"
            rows.append(r)
    pd.DataFrame(rows).to_csv(a.out/"v4_split_comparison.csv",index=False,encoding="utf-8-sig")

    rows=[]
    for y,z in c.groupby("year"):
        rows.append({"year":int(y),**summarize(z),"exchange":"ALL"})
        for ex,q in z.groupby("exchange"):
            rows.append({"year":int(y),**summarize(q),"exchange":ex})
    pd.DataFrame(rows).to_csv(a.out/"v4_yearly_comparison.csv",index=False,encoding="utf-8-sig")

    keys=c[["series_id","signal_date"]].drop_duplicates()
    ct=t.merge(keys,on=["series_id","signal_date"],how="inner")
    rows=[]
    for (sp,m),q in ct.groupby(["split","method"]):
        r={"split":sp,"method":m,**summarize(q,"net_return"),
           "stop_rate":float(q.stopped.mean()),"avg_hold_days":float(q.hold_days.mean())}
        r["evidence_status"]="POSTHOC_TEST" if sp=="TEST" else "DEVELOPMENT"
        rows.append(r)
    pd.DataFrame(rows).to_csv(a.out/"v4_stop_comparison.csv",index=False,encoding="utf-8-sig")

    manifest="Dante V4 locked candidate\n"+"\n".join(f"{k}={v}" for k,v in RULE.items())
    manifest+="\nNOTE: TEST was already inspected in V3.1; TEST rows are post-hoc confirmation only, not pristine OOS.\n"
    (a.out/"v4_rule_manifest.txt").write_text(manifest,encoding="utf-8")
    print(pd.read_csv(a.out/"v4_split_comparison.csv").to_string(index=False))

if __name__=="__main__":
    main()
