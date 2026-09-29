#!/usr/bin/env python3
"""Add point-in-time market-regime features to Dante signal outputs."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def load_panel(path: Path, ex: str) -> pd.DataFrame:
    x=pd.read_parquet(path,columns=["series_id","exchange","date","close","adjusted_close"]).copy()
    if "exchange" not in x: x["exchange"]=ex
    if "adjusted_close" not in x: x["adjusted_close"]=x["close"]
    x["date"]=pd.to_datetime(x["date"],errors="coerce")
    x["adjusted_close"]=pd.to_numeric(x["adjusted_close"],errors="coerce")
    return x.dropna(subset=["series_id","date","adjusted_close"]).sort_values(["series_id","date"])

def enrich(ko: Path, kq: Path, signals: pd.DataFrame):
    panel=pd.concat([load_panel(ko,"KO"),load_panel(kq,"KQ")],ignore_index=True)
    grp=panel.groupby(["exchange","series_id"],sort=False)["adjusted_close"]
    panel["stock_ret20"]=grp.pct_change(20,fill_method=None)
    panel["stock_ret60"]=grp.pct_change(60,fill_method=None)
    panel["up20"]=(panel.stock_ret20>0).where(panel.stock_ret20.notna())
    panel["up60"]=(panel.stock_ret60>0).where(panel.stock_ret60.notna())
    regime=panel.groupby(["exchange","date"],as_index=False).agg(
        market_breadth20=("up20","mean"), market_breadth60=("up60","mean"),
        market_median_ret20=("stock_ret20","median"), market_median_ret60=("stock_ret60","median"),
        market_n=("series_id","nunique"),
    ).sort_values(["exchange","date"])
    regime["market_breadth20_chg20"]=regime.groupby("exchange",sort=False).market_breadth20.diff(20)
    regime["market_breadth60_chg20"]=regime.groupby("exchange",sort=False).market_breadth60.diff(20)
    stock=panel[["exchange","series_id","date","stock_ret20","stock_ret60"]]
    s=signals.copy(); s["signal_date"]=pd.to_datetime(s.signal_date)
    s=s.merge(stock,left_on=["exchange","series_id","signal_date"],right_on=["exchange","series_id","date"],how="left").drop(columns=["date"])
    return s.merge(regime,left_on=["exchange","signal_date"],right_on=["exchange","date"],how="left").drop(columns=["date"])

def metric(q):
    x=pd.to_numeric(q.fwd60,errors="coerce").dropna()
    if x.empty: return {"n":0,"mean":np.nan,"median":np.nan,"trim10_mean":np.nan,"win_rate":np.nan}
    a=np.sort(x.to_numpy()); k=int(len(a)*.10); trim=a[k:len(a)-k] if len(a)>=10 and 2*k<len(a) else a
    return {"n":len(x),"mean":x.mean(),"median":x.median(),"trim10_mean":float(np.mean(trim)),"win_rate":(x>0).mean()}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ko",type=Path,required=True); ap.add_argument("--kq",type=Path,required=True)
    ap.add_argument("--signals",type=Path,required=True); ap.add_argument("--out",type=Path,default=Path("dante_results"))
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    s=enrich(a.ko,a.kq,pd.read_csv(a.signals))
    s["entry_gap"]=s.entry/s.ema224-1
    s["convergence_gap"]=(s.ema224-s.ema112)/s.ema224
    s["headroom448"]=s.ema448/s.entry-1
    s.to_csv(a.out/"signals_v5_regime.csv",index=False,encoding="utf-8-sig")
    common=(s.entry_gap<=.05)&(s.convergence_gap<=.05)&(s.headroom448>=.05)&(s.volume_ratio<=5)
    setups={"BASELINE":pd.Series(True,index=s.index),"V4_SIMPLE":common,"V4_STRICT":common&(s.below_80>=80)}
    regimes={
      "ALL_REGIMES":pd.Series(True,index=s.index),
      "BREADTH20_GE_50":s.market_breadth20>=.50,
      "BREADTH20_GE_60":s.market_breadth20>=.60,
      "BREADTH60_GE_50":s.market_breadth60>=.50,
      "BREADTH20_60_GE_50":(s.market_breadth20>=.50)&(s.market_breadth60>=.50),
      "BREADTH20_RISING":s.market_breadth20_chg20>0,
      "MEDIAN20_POSITIVE":s.market_median_ret20>0,
    }
    rows=[]
    for setup,sm in setups.items():
      for regime,rm in regimes.items():
        for sp,q in s[sm&rm].groupby("split"):
            r={"setup_rule":setup,"regime_rule":regime,"split":sp,**metric(q)}
            r["evidence_status"]="POSTHOC_TEST" if sp=="TEST" else "DEVELOPMENT"
            rows.append(r)
    out=pd.DataFrame(rows)
    out.to_csv(a.out/"v5_regime_comparison.csv",index=False,encoding="utf-8-sig")
    print(out.to_string(index=False))

if __name__=="__main__": main()
