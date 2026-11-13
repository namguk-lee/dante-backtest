#!/usr/bin/env python3
"""Paired as-of backtest: first EMA224 breakout vs pullback/support/rebreakout.

Research-only approximation of publicly discussed Dante structures. Uses the
SAME original EMA224-breakout parent signals from dante_base_v4_daily_backtest.
Counterfactual waiting strategy is cash (0% return) if no confirmation; also
reports filled-only comparison to disclose selection. Parents are deduplicated
per security and profile with a 60-session cooldown.

Baseline: V4 next-open, stop/2R within 20 or 60 sessions after signal.
Waiting: after breakout, 2-12 sessions later find price re-touch near EMA224
with lower volume and >=2.5% retreat from intervening high. In following 8
sessions, require 3-day (or 5-day) closing breakout, above EMA224/112, with
volume confirmation. Enter NEXT open only. Use a fresh pullback structural
stop and 2R target, exit by the SAME parent-horizon deadline. No same-bar
buy / sell, no use of future bars in signal predicates. Stop-first on candles
that cross both levels; gap-through executes at open. 40bp trading costs.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

PARENT_PROFILES=("B40_T50EOK", "B60_T50EOK", "B90_T50EOK")
VARIANTS={
    "REBREAK_3D":dict(rebreak=3,volratio=1.0,pull_vol=.90),
    "REBREAK_5D":dict(rebreak=5,volratio=1.2,pull_vol=.80),
}
HORIZONS=(20,60)
COOLDOWN=60
COST_PCT=.4
FIRST_PULL=2
LAST_PULL=12
MAX_REBREAK=8
ENTRY_GAP_MAX=.07
MIN_RISK=.025
MAX_RISK=.18


def fnum(x):
    try:
        return float(x)
    except (ValueError,TypeError):
        return np.nan


def arrays(g):
    g=g.sort_values("date").reset_index(drop=True)
    raw=g.close.to_numpy(dtype=float)
    adj=g.adjusted_close.to_numpy(dtype=float)
    ratio=np.divide(adj,raw,out=np.ones(len(raw)),where=raw>0)
    o=g.open.to_numpy(dtype=float)*ratio
    hi=g.high.to_numpy(dtype=float)*ratio
    lo=g.low.to_numpy(dtype=float)*ratio
    volume=g.volume.to_numpy(dtype=float)
    v20=pd.Series(volume).rolling(20).mean().to_numpy()
    ema112=pd.Series(adj).ewm(span=112,adjust=False,min_periods=112).mean().to_numpy()
    ema224=pd.Series(adj).ewm(span=224,adjust=False,min_periods=224).mean().to_numpy()
    return dict(g=g,c=adj,o=o,h=hi,l=lo,vol=volume,v20=v20,e112=ema112,e224=ema224,
                ix={str(d.date()):i for i,d in enumerate(pd.to_datetime(g.date))})


def wait_signal(d, i, variant):
    """Detect delayed pullback and re-breakout without looking past each bar."""
    c,o,hi,lo,vol,v20,e112,e224=(d[k] for k in ("c","o","h","l","vol","v20","e112","e224"))
    n=len(c)
    rule=VARIANTS[variant]
    if i+FIRST_PULL+2>=n:
        return None,"data_short"
    first_low=float(lo[i])
    for j in range(i+FIRST_PULL, min(n-2,i+LAST_PULL)+1):
        # This exit applies only to information known by the potential pull date.
        if not np.isfinite(e224[j]) or c[j]<e224[j]*.965 or c[j]<first_low*.94:
            return None,"support_lost"
        if not np.isfinite(v20[j]) or v20[j]<=0:
            continue
        peak=float(np.nanmax(hi[i:j+1]))
        pull_pct=c[j]/peak-1
        support=lo[j]<=e224[j]*1.035 and e224[j]*.985<=c[j]<=e224[j]*1.065
        quiet=vol[j]<=vol[i]*rule["pull_vol"] and vol[j]<=v20[j]*1.35
        retreat=-.22<=pull_pct<=-.025
        if not (support and quiet and retreat):
            continue
        for k in range(j+1,min(n-2,i+20,j+MAX_REBREAK)+1):
            if c[k]<e224[k]*.97 or lo[k]<lo[j]*.975:
                break
            if not all(np.isfinite(z) and z>0 for z in (v20[k],e112[k],e224[k],c[k])):
                continue
            old_hi=float(np.nanmax(hi[max(0,k-rule["rebreak"]):k]))
            if not(c[k]>old_hi and c[k]>=e224[k] and c[k]>=e112[k] and
                   vol[k]>=v20[k]*rule["volratio"] and c[k]>=o[k]):
                continue
            entry=float(o[k+1])
            if not (np.isfinite(entry) and entry>0 and entry<=c[k]*(1+ENTRY_GAP_MAX)):
                continue
            stop=max(float(np.nanmin(lo[j:k+1]))*.995, float(e224[k])*.965)
            risk=1-stop/entry
            if not(MIN_RISK<=risk<=MAX_RISK):
                continue
            return dict(pull_i=j,confirm_i=k,entry_i=k+1,entry=entry,stop=stop,
                        target=entry+2*(entry-stop),risk_pct=100*risk,
                        pull_to_confirm=k-j,from_signal_to_entry=k+1-i), "filled"
    return None,"no_confirmation"


def trade_until_parent_end(d, signal_i, plan, horizon):
    """At most horizon sessions after ORIGINAL signal, same deadline for both arms."""
    end=signal_i+horizon
    if end>=len(d["c"]):
        return None
    entry_i=int(plan["entry_i"])
    if entry_i>end:
        return None
    o,hi,lo,c=(d[k] for k in ("o","h","l","c"))
    entry,stop,target=(float(plan[k]) for k in ("entry","stop","target"))
    exit_price=float(c[end]);exit_i=end;outcome="TIMEOUT"
    for j in range(entry_i,end+1):
        vals=(o[j],hi[j],lo[j],c[j])
        if not all(np.isfinite(x) and x>0 for x in vals):
            return None
        if o[j]<=stop:
            exit_price=float(o[j]);exit_i=j;outcome="STOP_GAP";break
        if o[j]>=target:
            exit_price=float(o[j]);exit_i=j;outcome="TARGET_GAP";break
        if lo[j]<=stop:
            exit_price=stop;exit_i=j;outcome="STOP";break
        if hi[j]>=target:
            exit_price=target;exit_i=j;outcome="TARGET";break
    net=100*(exit_price/entry-1)-COST_PCT
    return dict(net_pct=net,outcome=outcome,exit_i=exit_i,
                exit_date=str(pd.Timestamp(d["g"].date.iloc[exit_i]).date()),
                exit_adj=exit_price)


def test_logic():
    # A constructed 224 breakout, quiet touch and renewed 3-day high.
    n=80
    c=np.full(n,100.);o=np.full(n,100.);hi=np.full(n,101.);lo=np.full(n,99.)
    vol=np.full(n,100.);v20=np.full(n,100.)
    i=40
    c[i]=105.;o[i]=101.;hi[i]=106.;lo[i]=100.;vol[i]=250.
    c[i+1]=103.;o[i+1]=104.;hi[i+1]=104.;lo[i+1]=102.
    c[i+2]=101.5;o[i+2]=102.;hi[i+2]=103.;lo[i+2]=99.8;vol[i+2]=70
    c[i+3]=101.2;o[i+3]=101.6;hi[i+3]=102;lo[i+3]=99.7;vol[i+3]=65
    c[i+4]=102.;o[i+4]=101.;hi[i+4]=102.3;lo[i+4]=100.7
    c[i+5]=103.8;o[i+5]=102.;hi[i+5]=104.;lo[i+5]=101;vol[i+5]=140.
    o[i+6]=104.;hi[i+6]=106.;lo[i+6]=102.;c[i+6]=105
    d=dict(c=c,o=o,h=hi,l=lo,vol=vol,v20=v20,
           e112=np.full(n,99.),e224=np.full(n,100.),
           g=pd.DataFrame({"date":pd.bdate_range("2024-01-01",periods=n)}))
    w,reason=wait_signal(d,i,"REBREAK_3D")
    assert reason=="filled" and w and w["entry_i"]==i+6, (reason,w)
    no=dict(d);no["vol"]=vol.copy();no["vol"][i+2:i+4]=500
    w2,_=wait_signal(no,i,"REBREAK_3D")
    assert w2 is None, "heavy-volume pullback incorrectly recognized"
    plan=dict(entry_i=i+6,entry=104.,stop=98.,target=116.)
    ret=trade_until_parent_end(d,i,plan,20)
    assert ret is not None and np.isfinite(ret["net_pct"])
    # Stop must execute at open, including a down gap.
    bad=dict(d);bad["o"]=o.copy();bad["o"][i+7]=90
    assert trade_until_parent_end(bad,i,plan,20)["outcome"]=="STOP_GAP"
    print("synthetic tests passed",flush=True)


def compare_rows(events,market):
    events=events[
        events.profile.isin(PARENT_PROFILES) & events.setup.eq("EMA224_BREAKOUT")
    ].copy()
    events["date"]=pd.to_datetime(events.date)
    events=events.sort_values(["date","code","profile"])
    # Deduplicate by ticker within EACH profile and its corresponding trading episode;
    # unlike V4, no overlap from different event types enters the paired sample.
    out=[]
    present=set()
    for group_id,g in market.groupby("series_id",sort=False):
        code=str(g.code.iloc[-1]).zfill(6)
        event_group=events[(events.code.eq(code)) &
                           (events.date.ge(g.date.min())) &
                           (events.date.le(g.date.max()))]
        if event_group.empty:
            continue
        d=arrays(g)
        by_profile={v:-100000 for v in PARENT_PROFILES}
        for ev in event_group.sort_values(["date","profile"]).itertuples(index=False):
            p=str(ev.profile)
            dt=str(pd.Timestamp(ev.date).date())
            i=d["ix"].get(dt)
            if i is None:
                continue
            if i-by_profile[p]<COOLDOWN:
                continue
            by_profile[p]=i
            signal_name=str(g.name.iloc[i])
            # Reject ticker reuse where historical identity differs.
            if signal_name!=str(ev.name):
                continue
            for variant in VARIANTS:
                plan,why=wait_signal(d,i,variant)
                row=dict(parent_date=dt,year=int(pd.Timestamp(dt).year),
                         code=code,name=signal_name,exchange=str(g.exchange.iloc[i]),
                         episode=str(group_id),profile=p,variant=variant,
                         base_days=int(ev.base_days),
                         signal_close=float(ev.signal_close_raw),
                         baseline_entry=float(ev.entry_open_adjusted),
                         baseline_stop=float(ev.stop_adjusted),
                         signal_ema224=float(d["e224"][i]),
                         signal_vol_ratio=float(ev.vol_ratio),
                         wait_status=why,wait_filled=plan is not None,
                         wait_pull_date=(str(pd.Timestamp(g.date.iloc[plan["pull_i"]]).date()) if plan else None),
                         wait_confirm_date=(str(pd.Timestamp(g.date.iloc[plan["confirm_i"]]).date()) if plan else None),
                         wait_entry_date=(str(pd.Timestamp(g.date.iloc[plan["entry_i"]]).date()) if plan else None),
                         wait_entry=(plan["entry"] if plan else np.nan),
                         wait_stop=(plan["stop"] if plan else np.nan),
                         wait_target=(plan["target"] if plan else np.nan),
                         wait_risk_pct=(plan["risk_pct"] if plan else np.nan),
                         wait_delay=(plan["from_signal_to_entry"] if plan else np.nan))
                for h in HORIZONS:
                    mature=bool(i+h<len(d["c"]))
                    base_net=fnum(getattr(ev,f"net_{h}d"))
                    row[f"mature_{h}d"]=mature and np.isfinite(base_net)
                    row[f"base_net_{h}d"]=(base_net if row[f"mature_{h}d"] else np.nan)
                    row[f"base_outcome_{h}d"]=(getattr(ev,f"outcome_{h}d")
                                                if row[f"mature_{h}d"] else "")
                    if row[f"mature_{h}d"]:
                        # Strategy with no fill parks cash at 0%; no fee is paid.
                        if plan is None or plan["entry_i"]>i+h:
                            row[f"wait_itt_net_{h}d"]=0.0
                            row[f"wait_traded_{h}d"]=False
                            row[f"wait_outcome_{h}d"]="NO_ENTRY"
                        else:
                            tr=trade_until_parent_end(d,i,plan,h)
                            row[f"wait_itt_net_{h}d"]=(tr["net_pct"] if tr else np.nan)
                            row[f"wait_traded_{h}d"]=tr is not None
                            row[f"wait_outcome_{h}d"]=(tr["outcome"] if tr else "DATA_ERROR")
                            row[f"wait_exit_{h}d"]=(tr["exit_date"] if tr else "")
                    else:
                        row[f"wait_itt_net_{h}d"]=np.nan
                        row[f"wait_traded_{h}d"]=False
                        row[f"wait_outcome_{h}d"]="CENSORED"
                out.append(row)
        present.add(str(group_id))
    return pd.DataFrame(out)


def stats(q,keys):
    rows=[]
    for vals,g in q.groupby(keys,dropna=False):
        if not isinstance(vals,tuple):vals=(vals,)
        r=dict(zip(keys,vals))
        r.update(parents=len(g),unique_stocks=g.code.nunique(),
                 confirmed=int(g.wait_filled.sum()),
                 confirmed_pct=100*g.wait_filled.mean(),
                 median_confirmation_delay=float(g.wait_delay.median())
                     if g.wait_filled.any() else np.nan)
        for h in HORIZONS:
            m=g[g[f"mature_{h}d"].eq(True)].copy()
            b=m[f"base_net_{h}d"].astype(float)
            w=m[f"wait_itt_net_{h}d"].astype(float)
            valid=b.notna()&w.notna()
            m=m[valid];b=b[valid];w=w[valid]
            filled=m[m[f"wait_traded_{h}d"].eq(True)]
            r.update({f"mature_{h}d":len(m),
                      f"traded_{h}d":len(filled),
                      f"fill_pct_{h}d":100*len(filled)/len(m) if len(m) else np.nan,
                      f"base_mean_{h}d":b.mean() if len(m) else np.nan,
                      f"wait_itt_mean_{h}d":w.mean() if len(m) else np.nan,
                      f"delta_mean_{h}d":(w-b).mean() if len(m) else np.nan,
                      f"base_median_{h}d":b.median() if len(m) else np.nan,
                      f"wait_itt_median_{h}d":w.median() if len(m) else np.nan,
                      f"base_win_pct_{h}d":100*(b>0).mean() if len(m) else np.nan,
                      f"wait_win_pct_{h}d":100*(w>0).mean() if len(m) else np.nan,
                      f"wait_filled_mean_{h}d":filled[f"wait_itt_net_{h}d"].mean()
                           if len(filled) else np.nan,
                      f"base_filled_mean_{h}d":filled[f"base_net_{h}d"].mean()
                           if len(filled) else np.nan,
                      f"wait_stop_pct_{h}d":100*filled[f"wait_outcome_{h}d"].astype(str).str.startswith("STOP").mean()
                           if len(filled) else np.nan})
        rows.append(r)
    return pd.DataFrame(rows)


def test_pair_guard():
    # A single parent per profile in a 60-trading-day window is expected.
    fake=pd.DataFrame([dict(profile="x",date=d) for d in (1,40,61,100)])
    chosen=[];last=-100000
    for i in fake.date:
        if i-last<COOLDOWN:continue
        chosen.append(i);last=i
    assert chosen==[1,61]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--ko",type=Path,required=True)
    p.add_argument("--kq",type=Path,required=True)
    p.add_argument("--parent",type=Path,required=True,
                   help="Input V4 signals.csv regenerated with pinned panel")
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    test_logic();test_pair_guard()
    a.out.mkdir(exist_ok=True,parents=True)
    from dante_classic_scanner import load
    market=pd.concat([load(a.ko,"KO"),load(a.kq,"KQ")],ignore_index=True)
    events=pd.read_csv(a.parent,dtype={"code":str},encoding="utf-8-sig")
    print("BASELINE EVENTS",len(events),"PANEL",len(market),flush=True)
    df=compare_rows(events,market)
    if df.empty:
        raise RuntimeError("No paired breakout events; inspect input panel and parent CSV")
    df["split"]=np.select([df.year<=2021,df.year<=2024],["TRAIN","VALID"],default="TEST")
    df=df.sort_values(["parent_date","code","profile","variant"])
    df.to_csv(a.out/"paired_trades.csv",index=False,encoding="utf-8-sig")
    tables=[
        ("summary_by_split.csv",["profile","variant","split"]),
        ("summary_by_year.csv",["profile","variant","year"]),
        ("summary_by_variant.csv",["profile","variant"]),
        ("summary_by_market.csv",["profile","variant","split","exchange"]),
    ]
    for filename,keys in tables:
        stats(df,keys).to_csv(a.out/filename,index=False,encoding="utf-8-sig")
    summary=stats(df,["profile","variant","split"])
    print("\n=== PAIRED COMPARISON WITH FULL-CASH NON-FILL ===\n",summary.to_string(index=False),flush=True)
    print("\n=== YEARLY FILLED / SOURCE PARENTS ===\n",
          stats(df,["profile","variant","year"])[["profile","variant","year",
                    "parents","confirmed","mature_20d","traded_20d",
                    "base_mean_20d","wait_itt_mean_20d","delta_mean_20d"]].to_string(index=False),
          flush=True)
    # Prevent promoting cherry-picked positive conditional P&L.
    print("WARNING: conditional filled-only performance is selection-biased. "
          "Primary metric is same-cohort ITT, nonfills cash=0, common signal-date horizon.",flush=True)
    print("WARNING: adjusted prices for corporate actions are approximate. "
          "Not live trading or an untouched OOS set. No trading orders.",flush=True)


if __name__=="__main__":
    main()
