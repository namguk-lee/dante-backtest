#!/usr/bin/env python3
"""Compare EXACT SAME technical price/volume events under legacy vs cap-normalized gating.

Research-only sensitivity benchmark, not a financial recommendation.
Prior20 amount/cap must be known at signal close, not sourced from future.
Benchmark outcome = next adjusted open -> 20-session adjusted close minus 40bp,
which differs from 2R/stop trade strategies. Mature results only.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from cap_liquidity import attach_liquidity_features

FEE_PCT=.4

def run_episode(g):
    z=g.sort_values("date").reset_index(drop=True).copy()
    if len(z)<70:return pd.DataFrame(),{}
    factor=(z.adjusted_close/z.close).replace([np.inf,-np.inf],np.nan).ffill().bfill()
    op=z.open.astype(float)*factor
    cl=z.adjusted_close.astype(float)
    z=attach_liquidity_features(z)
    # Match the original event-first stage-1 technical conditions exactly,
    # with today's volume included in legacy 20-day volume mean. This study
    # changes ONLY the liquidity gate, not the technical event denominator.
    v20=z.volume.rolling(20,min_periods=20).mean()
    ratio=z.volume/v20
    ret=cl.pct_change(fill_method=None)
    body=cl/op-1
    strong=ret.ge(.05)&body.ge(.035)&ratio.ge(2.0)
    idx=np.flatnonzero(strong.fillna(False).to_numpy())
    if not len(idx):return pd.DataFrame(),{}
    a=z.iloc[idx].copy()
    a["volume_multiple_legacy"]=ratio.iloc[idx].to_numpy()
    a["event_return_pct"]=100*ret.iloc[idx].to_numpy()
    a["event_body_pct"]=100*body.iloc[idx].to_numpy()
    a["legacy"]=a.legacy_50e_pass.astype(bool)
    a["cap_relative"]=a.adaptive_liquidity_pass.astype(bool)
    a["label"]=np.select(
        [a.legacy&a.cap_relative,~a.legacy&a.cap_relative,
         a.legacy&(~a.cap_relative)],["BOTH","CAP_ONLY","LEGACY_ONLY"],
         default="NEITHER")
    # Historical 20d return only when future values are actually present.
    forward_idx=idx+20
    good=forward_idx<len(z)
    entry=np.full(len(idx),np.nan)
    exit=np.full(len(idx),np.nan)
    entry[good]=op.to_numpy()[idx[good]+1]
    exit[good]=cl.to_numpy()[forward_idx[good]]
    a["mature20"]=good
    a["hold20_net_pct"]=np.where((entry>0)&(exit>0),
          100*(exit/np.where(entry>0,entry,np.nan)-1)-FEE_PCT,np.nan)
    a["year"]=a.date.dt.year
    # Dates refer to same issuer-episode. No time-series merging across re-list.
    a["signal_date"]=a.date.dt.strftime("%Y-%m-%d")
    keep=["signal_date","year","code","name","exchange","series_id",
          "cap_tier","market_cap_quality","market_cap_won",
          "pre20_amount_won","relative_turnover_pre20_pct",
          "daily_turnover_to_cap_pct","listed_share_volume_turnover_pct",
          "amount_spike_vs_prior20","volume_multiple_legacy",
          "event_return_pct","event_body_pct","liquidity_warning",
          "legacy","cap_relative","label","mature20","hold20_net_pct"]
    counts={
        "latest_price_date":str(z.date.iloc[-1].date()),
        "rawcap_won_checked":int(z.market_cap_quality.eq("KRX_MARCAP_WON_CHECKED").sum()),
        "rawcap_million_checked":int(z.market_cap_quality.eq("MARCAP_MILLION_CONVERTED_CHECKED").sum()),
        "computed_estimate":int(z.market_cap_quality.isin(["CLOSE_X_LISTED_SHARES_ESTIMATE","LAST_REPORTED_SHARES_ESTIMATE","MARCAP_MISMATCH_ESTIMATE"]).sum()),
        "missing_cap":int(z.market_cap_quality.eq("CAP_MISSING").sum()),
        "raw_source_conflicts":int(z.cap_reported_conflict.sum())
    }
    return a[keep],counts

def summarize(d):
    rows=[]
    d["split"]=np.select([d.year.le(2021),d.year.le(2024)],
                         ["TRAIN_2018_21","VALID_2022_24"],default="LATER_2025_26")
    for split,g in [*d.groupby("split"),("ALL",d)]:
        for tier,gg in [*g.groupby("cap_tier"),("ALL_CAP",g)]:
            for subset,mask in {
                "LEGACY_50E":gg.legacy,
                "CAP_RELATIVE":gg.cap_relative,
                "BOTH":gg.label.eq("BOTH"),
                "CAP_ONLY":gg.label.eq("CAP_ONLY"),
                "LEGACY_ONLY":gg.label.eq("LEGACY_ONLY"),
                "NEITHER":gg.label.eq("NEITHER"),
            }.items():
                q=gg[mask];m=q.dropna(subset=["hold20_net_pct"])
                rows.append(dict(split=split,cap_tier=tier,method=subset,
                     signal_rows=int(len(q)),distinct_issuers=int(q.code.nunique()),
                     matured20=int(len(m)),mean_20d_net_pct=float(m.hold20_net_pct.mean()),
                     median_20d_net_pct=float(m.hold20_net_pct.median()),
                     positive_20d_pct=float(100*m.hold20_net_pct.gt(0).mean()) if len(m) else np.nan,
                     median_pre20_amount_billion=float(q.pre20_amount_won.median()/1e9) if len(q) else np.nan,
                     median_relative_cap_turnover_pct=float(q.relative_turnover_pre20_pct.median()) if len(q) else np.nan))
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--ko",type=Path,required=True)
    p.add_argument("--kq",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    from dante_classic_scanner import load
    panel=pd.concat([load(args.ko,"KO"),load(args.kq,"KQ")],ignore_index=True)
    frames=[];qualities=[]
    for n,(sid,g) in enumerate(panel.groupby("series_id",sort=False),1):
        t,q=run_episode(g)
        if not t.empty:frames.append(t)
        if q:qualities.append(q)
        if n%500==0:print("issuer_episode",n,"eligible_technical_events",sum(map(len,frames)),flush=True)
    if not frames:raise RuntimeError("No strong daily event rows; test source panel")
    e=pd.concat(frames,ignore_index=True)
    e=e[e.year.ge(2018)]
    e.to_csv(args.out/"all_technical_events_with_liquidity.csv",index=False,encoding="utf-8-sig")
    s=summarize(e)
    s.to_csv(args.out/"comparative_event_screen_outcomes.csv",index=False,encoding="utf-8-sig")
    wk=e[e.signal_date.between("2026-11-09","2026-11-13")].copy()
    wk.sort_values(["signal_date","label","code"]).to_csv(
        args.out/"recent_week_screen_difference.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(qualities).sum(numeric_only=True).to_frame("price_bar_rows").to_csv(
        args.out/"cap_source_quality_audit.csv",encoding="utf-8-sig")
    print("SOURCE_ROWS",len(panel),"ELIGIBLE_TECH_EVENTS",len(e),"RECENT_WEEK",len(wk),flush=True)
    print("WEEK LABELS\n"+wk.label.value_counts().to_string(),flush=True)
    print("SPLITS\n"+s[s.cap_tier.eq("ALL_CAP")].to_string(index=False),flush=True)
    print("CAP SOURCE CHECK\n"+pd.DataFrame(qualities).sum(numeric_only=True).to_string(),flush=True)
    (args.out/"METHOD.txt").write_text(
        "Research only. Exactly identical initial technical 5% return, 3.5% body, "
        "2x (volume/current-inclusive 20D mean) screen under two distinct "
        "amount gates: legacy 50e KRW 20D average including signal candle, "
        "cap relative 20D average EXCLUDING signal candle with previous close "
        "market capitalization and 3-size-tier heuristic. Not a finished "
        "Dante classic/EMA224 signal. Outcomes: following day's adjusted open "
        "to +20 trading-day adjusted close, minus 0.4 percentage points; "
        "NO stop/2R. Newly eligible and rejected names may differ by "
        "date/model; overlapping events can be correlated and cannot be "
        "treated as independent. Missing cap cannot pass cap-relative. "
        "Not a free-float-adjusted ratio. Source cap units cross-checked "
        "against raw close x listed shares. Historical periods have previously "
        "been examined, so validation is exploratory. No trading orders.\n",
        encoding="utf-8")
if __name__=="__main__":main()
