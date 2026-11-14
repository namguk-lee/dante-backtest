#!/usr/bin/env python3
"""Stage 4: V4 full-market 2R breakout success vs failure.

RESEARCH ONLY. As-of-date predictive features exclude next-session entry/stop
risk. Targets, stops and gap-aware outcomes are inherited from V4. Six (or
more, on other snapshots) immature/censored rows MUST NOT become TIMEOUT.
All periods have been inspected in prior project research: this is NOT a
pristine prospective holdout, even when labeled TRAIN/VALID/TEST.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

FEATURES=["base_range_pct","base_change_pct","prior_peak_drop_pct",
          "dist224_pct","vol_ratio","below224_of_last80"]
SPLITS=("TRAIN","VALID","TEST")

def prepare(path):
    d=pd.read_csv(path,dtype={"code":str},low_memory=False)
    d=d[d.profile.eq("B60_T50EOK") & d.setup.eq("EMA224_BREAKOUT")].copy()
    d["date"]=pd.to_datetime(d.date)
    d["mature"]=d.net_20d.notna()&d.outcome_20d.notna()
    o=d.outcome_20d.fillna("")
    d["path_outcome"]=np.select([o.str.startswith("TARGET"),o.str.startswith("STOP"),o.eq("TIMEOUT")],
                                ["TARGET","STOP","TIMEOUT"],default="CENSORED")
    d["target_hit"]=d.path_outcome.eq("TARGET").astype(int)
    d["stable"]=(d.base_range_pct<=25)&d.base_change_pct.between(-5,15)
    d["volume2"]=d.vol_ratio>=2
    d["near224"]=d.dist224_pct.between(-1,5)
    d["combo"]=d.stable&d.volume2&d.near224
    last={};keep={}
    for i,r in d.sort_values(["date","code"]).iterrows():
        old=last.get(r.code)
        ok=old is None or (r.date-old).days>=90
        keep[i]=ok
        if ok:last[r.code]=r.date
    d["cooldown90calendar"]=d.index.map(keep).fillna(False).astype(bool)
    return d

def cohort_table(d):
    result=[]
    for use_cooldown in (False,True):
        c=d[d.cooldown90calendar] if use_cooldown else d
        for phase in (*SPLITS,"ALL"):
            p=c[c.split.eq(phase)] if phase!="ALL" else c
            for label,mask in {
                "original":pd.Series(True,index=p.index),
                "base_stable":p.stable,
                "volume2":p.volume2,
                "near224":p.near224,
                "combined":p.combo,
                "combined_with_next_open_risk_le8_DIAGNOSTIC_ONLY":p.combo & p.stop_risk_pct.le(8)
            }.items():
                z=p[mask];m=z[z.mature]
                if z.empty:continue
                counts=m.path_outcome.value_counts()
                result.append(dict(
                    split=phase,group=label,cooldown90calendar=use_cooldown,
                    total=len(z),mature=len(m),censored=len(z)-len(m),
                    unique_codes=z.code.nunique(),
                    target=int(counts.get("TARGET",0)),
                    stop=int(counts.get("STOP",0)),
                    timeout=int(counts.get("TIMEOUT",0)),
                    target_rate_pct=100*m.target_hit.mean() if len(m) else np.nan,
                    stop_rate_pct=100*m.path_outcome.eq("STOP").mean() if len(m) else np.nan,
                    mean_net20_pct=m.net_20d.mean(),median_net20_pct=m.net_20d.median()))
    return pd.DataFrame(result)

def issuer_bootstrap(d):
    rng=np.random.default_rng(20261114)
    rows=[]
    for phase in SPLITS:
        x=d[d.mature & d.split.eq(phase)]
        if x.empty:continue
        g=x.groupby("code",sort=False).agg(n=("target_hit","size"),target=("target_hit","sum"),
                                            combo_n=("combo","sum"))
        g["combo_target"]=x[x.combo].groupby("code").target_hit.sum().reindex(g.index,fill_value=0)
        v=g[["n","target","combo_n","combo_target"]].to_numpy(float)
        draw=rng.integers(0,len(v),size=(1000,len(v)))
        z=v[draw].sum(axis=1);z=z[z[:,2]>=5]
        uplift=100*(z[:,3]/z[:,2]-z[:,1]/z[:,0])
        rows.append(dict(
            split=phase,total=len(x),combo=int(x.combo.sum()),
            target_rate_lift_pp=100*(x.loc[x.combo,"target_hit"].mean()-x.target_hit.mean()),
            cluster95_low=np.percentile(uplift,2.5),
            cluster95_high=np.percentile(uplift,97.5)))
    return pd.DataFrame(rows)

def train_fixed_model(d):
    s=d[d.mature].copy()
    features=s[FEATURES].copy()
    features["vol_ratio"]=np.log1p(features.vol_ratio.clip(lower=0))
    y=s.target_hit.astype(int)
    train=s.split.eq("TRAIN")
    clf=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                      LogisticRegression(C=.5,solver="lbfgs",max_iter=2000))
    clf.fit(features.loc[train],y.loc[train])
    s["pred_prob"]=clf.predict_proba(features)[:,1]
    boundary=float(np.quantile(s.loc[train,"pred_prob"],.80))
    res=[]
    for phase in SPLITS:
        x=s[s.split.eq(phase)]
        if x.empty:continue
        selected=x[x.pred_prob>=boundary]
        res.append(dict(
            split=phase,mature=len(x),target_rate_pct=100*x.target_hit.mean(),
            auc=roc_auc_score(x.target_hit,x.pred_prob),
            average_precision=average_precision_score(x.target_hit,x.pred_prob),
            brier=brier_score_loss(x.target_hit,x.pred_prob),
            train80_boundary=boundary,selected=len(selected),
            selected_target_pct=100*selected.target_hit.mean() if len(selected) else np.nan,
            selected_net20_mean=selected.net_20d.mean(),
            baseline_net20_mean=x.net_20d.mean(),
            baseline_net20_median=x.net_20d.median()))
    coefficients=pd.DataFrame({"feature":FEATURES,
        "standardized_coefficient":clf.named_steps["logisticregression"].coef_[0]})
    return pd.DataFrame(res),coefficients,s

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--signals",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    d=prepare(a.signals)
    cohort=cohort_table(d)
    model,coef,pred=train_fixed_model(d)
    boot=issuer_bootstrap(d)
    files={
        "v4_2r_success_failure_cohorts.csv":cohort,
        "fixed_model_time_split_metrics.csv":model,
        "model_coefficients.csv":coef,
        "cluster_bootstrap_research.csv":boot
    }
    for file,df in files.items():df.to_csv(a.out/file,index=False,encoding="utf-8-sig")
    pred[["date","code","name","split","path_outcome","target_hit",
          "net_20d","pred_prob","combo","cooldown90calendar"]].to_csv(
        a.out/"every_mature_breakout_prediction.csv",index=False,encoding="utf-8-sig")
    (a.out/"METHODOLOGY.txt").write_text(
        "As-of predictors: base price stability, prior decline, distance EMA224, "
        "signal relative volume and prior80 days below EMA224. No entry-next-open "
        "or future features in predictor. 20d V4 simulation applies real next "
        "OPEN adjusted entry, structural stop and 2R target with gap-aware and "
        "stop-first intraday collision; 40bp transaction cost already in V4. "
        "Immature outcomes excluded, NOT assigned TIMEOUT. "
        "2018-21 TRAIN, 2022-24 VALID, 2025-26 TEST only time partitions, "
        "NOT newly untouched OOS, because past research already observed data. "
        "1,000 issuer-cluster bootstrap replicates on matched nested filter "
        "target-rate uplift. 90-calendar-day pseudo cooldown is approx, not "
        "the same as 60 trading days. No original paid indicator formulas "
        "or automatic trading. No deployment to main.\n",encoding="utf-8")
    print("COHORTS\n",cohort[cohort.group.isin(["original","combined"]) & ~cohort.cooldown90calendar].to_string(index=False),flush=True)
    print("MODEL\n",model.to_string(index=False),flush=True)
    print("BOOTSTRAP\n",boot.to_string(index=False),flush=True)

if __name__=="__main__":main()
