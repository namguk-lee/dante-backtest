#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""As-of official-case STATE vs EVENT audit with same-date market controls.

Design protections:
- An officially discussed setup is NOT equivalent to a vendor signal on that date.
- Market controls are UNLABELED, NOT negative paid-Dante screen outputs.
- Every feature uses candles on or before official analysis date (no future OHLCV).
- Case-specific low-liquidity issues are shown, never silently excluded.
- Report concordance and market prevalence, never proprietary recall/precision.
- No trade orders, no production promotion.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

from dante_classic_scanner import load
from dante_independent_indicator_suite import prepare, event_tests
from dante_official_example_replay import CASES, direct_naver_history, MAP

DISCARD_LABELS={"EARLIER_HISTORICAL","FUTURE_POSSIBILITY"}
FAMILY_STATE={
    "ODORI":"odori_turn_recent20_state",
    "EMA112":"ema112_held_state",
    "REVERSE_BASE":"reverse_accum_base_state",
    "BOWL_BASE":"bowl_base_state",
    "ANCHOR":"anchor_held_state",
}

def finite(v):
    return pd.notna(v) and np.isfinite(float(v))

def get_feature_window(z,i):
    """Build independent structural states as of close i; no future bars."""
    q=z.iloc[:i+1]
    c=q.iloc[-1]
    def recent(mask,window):
        return bool(mask.iloc[max(0,len(q)-window):].fillna(False).any())
    def count_recent(mask,window):
        return int(mask.iloc[max(0,len(q)-window):].fillna(False).sum())
    adj_close=q.ac.astype(float)
    e5up=q.ema5>q.ema5.shift(1)
    turn5=e5up & ~(e5up.shift(1).fillna(False).astype(bool))
    sm5up=q.sma5>q.sma5.shift(1)
    sm5turn=sm5up & ~(sm5up.shift(1).fillna(False).astype(bool))
    close_above5=bool(finite(c.ema5) and c.ac>=c.ema5)
    close_above112=bool(finite(c.ema112) and c.ac>=c.ema112)
    close_above224=bool(finite(c.ema224) and c.ac>=c.ema224)
    rev=(q.ema112<q.ema224)&(q.ema224<q.ema448)
    crossover112=(q.ac>=q.ema112)&(q.ac.shift(1)<q.ema112.shift(1))
    crossover224=(q.ac>=q.ema224)&(q.ac.shift(1)<q.ema224.shift(1))
    anchor=(q.volratio>=1.8)&(q.body>=.025)&(q.closepos>=.65)&(q.ac>q.ao)
    anchor_recent=q[anchor & q.index.to_series().ge(max(0,i-120))]
    anchor_held=bool(len(anchor_recent) and c.ac>=anchor_recent.ao.iloc[-1])
    anchor_date=str(anchor_recent.date.iloc[-1].date()) if len(anchor_recent) else ""
    # Fixed, pre-declared research-only shape guard. Check base windows ending
    # on any of last 30 trading bars; allow breakout to occur after a base.
    stable40=False;stable60=False;best_range=np.nan;stable_end=""
    for j in range(max(80,i-29),i+1,5):
        for days in (40,60,90):
            if j-days<20:continue
            d=q.ac.iloc[j-days:j].to_numpy(dtype=float)
            if len(d)<days or not np.isfinite(d).all() or np.min(d)<=0:continue
            lower,upper=np.quantile(d,[.10,.90])
            width=float(upper/lower-1)
            net=float(d[-1]/d[0]-1)
            if width<=.30 and -.18<=net<=.20:
                stable40=True
                stable60=stable60 or days>=60
                if not finite(best_range) or width<best_range:
                    best_range=width;stable_end=str(q.date.iloc[j-1].date())
    last260=q.iloc[max(0,len(q)-260):]
    high=float(last260.ah.max()) if len(last260) else np.nan
    low=float(last260.al.min()) if len(last260) else np.nan
    drawdown=100*(low/high-1) if finite(high) and high>0 else np.nan
    dist224=100*(c.ac/c.ema224-1) if finite(c.ema224) and c.ema224>0 else np.nan
    rev_recent120=recent(rev,120)
    anchor_recent120=bool(len(anchor_recent))
    ema112_held=(close_above112 and recent(crossover112,80))
    odori20=(close_above5 and (recent(turn5,20) or recent(sm5turn,20)))
    # A very strict REVERSE+ANCHOR+BASE state; few may qualify.
    reverse_accum_base=bool(rev_recent120 and anchor_recent120 and stable40 and
                           (close_above112 or c.ac>=c.ema60))
    bowl_state=bool(stable40 and finite(drawdown) and drawdown<=-20 and
                    finite(dist224) and -15<=dist224<=20)
    return {
        "odori_turn_recent20_state":odori20,
        "ema112_held_state":ema112_held,
        "reverse_accum_base_state":reverse_accum_base,
        "bowl_base_state":bowl_state,
        "anchor_held_state":anchor_held,
        "reverse_array_in_prior120":rev_recent120,
        "reverse_array_now":bool(rev.iloc[-1]) if pd.notna(rev.iloc[-1]) else False,
        "ema112_above_now":close_above112,
        "ema224_above_now":close_above224,
        "ema112_cross_recent80":recent(crossover112,80),
        "ema224_cross_recent80":recent(crossover224,80),
        "ema5_turn_recent20":recent(turn5,20),
        "sma5_turn_recent20":recent(sm5turn,20),
        "anchor_candle_count120":count_recent(anchor,120),
        "anchor_held":anchor_held,
        "anchor_date":anchor_date,
        "stable_base40_recent30":stable40,
        "stable_base60_recent30":stable60,
        "narrowest_base_width_pct":float(best_range*100) if finite(best_range) else np.nan,
        "stable_base_end":stable_end,
        "past260_peak_to_trough_pct":drawdown,
        "price_to_ema224_pct":dist224,
        "avg_turnover_prior20_krw":float(c.preamount20) if finite(c.preamount20) else np.nan,
        "recent_volratio":float(c.volratio) if finite(c.volratio) else np.nan,
        "ema112_slope20_pct":float((c.ema112/q.ema112.iloc[-21]-1)*100) if len(q)>21 and finite(q.ema112.iloc[-21]) else np.nan,
    }

def canonical_row(g,asof,family,source,issuer=""):
    z=prepare(g)
    z=z[z.date.le(pd.Timestamp(asof))].reset_index(drop=True)
    if z.empty:
        return dict(status="EMPTY_HISTORY")
    age=(pd.Timestamp(asof)-z.date.iloc[-1]).days
    if age>7:return dict(status="STALE_DATA",age_calendar_days=age)
    if len(z)<448:
        return dict(status="NOT_ENOUGH_FOR_EMA448",history_bars=len(z))
    st=get_feature_window(z,len(z)-1)
    # Signal models require 560 valid bars; lower-history cases are still
    # assessed for observable setup STATE, never counted as event mismatch.
    event_families=set()
    if len(z)>=560:
        for j in range(max(560,len(z)-21),len(z)):
            for model,desc,base in event_tests(z,j):
                event_families.add(model)
    target=FAMILY_STATE.get(family)
    st.update({
        "status":"REPLAY_READY",
        "source":source,"code":str(z.code.iloc[-1]).zfill(6),
        "name":str(z.name.iloc[-1]),"date":str(pd.Timestamp(asof).date()),
        "price_date":str(z.date.iloc[-1].date()),
        "history_bars":len(z),"case_family":family,
        "target_state":bool(st[target]) if target else None,
        "candidate_event_last20":any(model in MAP[family] for model in event_families) if family in MAP else None,
        "all_proxy_event_names_last20":";".join(sorted(event_families)),
        "turnover_ge_5e9":bool(st["avg_turnover_prior20_krw"]>=5e9),
        "turnover_ge_1e9":bool(st["avg_turnover_prior20_krw"]>=1e9),
        "turnover_ge_5e8":bool(st["avg_turnover_prior20_krw"]>=5e8),
    })
    return st

def case_history(panel,index,case):
    code,name,asof,kind,label,key,description=case
    sub=panel[(panel.code.eq(code))&(panel.date.le(asof))]
    if sub.empty:
        return None,"NO_EPISODE"
    r=sub.iloc[-1];sid=r.series_id
    h=panel.iloc[index[str(sid)]].copy()
    if (pd.Timestamp(asof)-pd.Timestamp(r.date)).days>7:
        try:return direct_naver_history(code,name,asof),"FDR_NAVER_RECOVERY"
        except Exception as exc:
            return None,"RECOVERY_FAILED:"+str(exc)[:100]
    return h,"MARCAP"

def sha_order(day,code,seed="STAGE3_SAME_DATE_20261114"):
    return hashlib.sha256(f"{seed}/{day}/{code}".encode("utf-8")).hexdigest()

def market_controls(panel,index,case,subject_status,excluded,max_controls=30):
    day=case[2];casecode=case[0];kind=case[3]
    target_turnover=subject_status.get("avg_turnover_prior20_krw",np.nan)
    if not finite(target_turnover):target_turnover=5e8
    # Same actual close-day and same KOSPI/KOSDAQ board if available. Because
    # delisted/illiquid names can have odd histories, do not require exact
    # liquidity match; actual 20d liquidity match is a post-computation check.
    day_panel=panel[panel.date.eq(pd.Timestamp(day))]
    s_exchange=subject_status.get("exchange_code")
    if s_exchange:
        day_panel=day_panel[day_panel.exchange.eq(s_exchange)]
    candidates=day_panel[~day_panel.code.isin(excluded)].copy()
    candidates["hash"]=candidates.code.map(lambda v:sha_order(day,v))
    candidates=candidates.sort_values("hash")
    controls=[];weak=[]
    for row in candidates.itertuples(index=False):
        if len(controls)>=max_controls:break
        if len(controls)+len(weak)>=max_controls*6:break
        if str(row.series_id) not in index:continue
        hist=panel.iloc[index[str(row.series_id)]]
        if len(hist)<448:continue
        outcome=canonical_row(hist,day,kind,"MATCHED_CONTROL_UNLABELED")
        if outcome.get("status")!="REPLAY_READY":continue
        amount=outcome["avg_turnover_prior20_krw"]
        if not finite(amount):continue
        # Same board and broad +/- one log unit of 20d average KRW turnover.
        ratio=float(amount/target_turnover) if target_turnover>0 else np.nan
        if .10<=ratio<=10:
            outcome["match_turnover_ratio"]=ratio
            controls.append(outcome)
        else:
            weak.append(outcome)
    return controls

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--ko",type=Path,required=True)
    p.add_argument("--kq",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--controls-per-case",type=int,default=20)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    panel=pd.concat([load(args.ko,"KO"),load(args.kq,"KQ")],ignore_index=True)
    panel["code"]=panel.code.astype(str).str.zfill(6)
    panel["date"]=pd.to_datetime(panel.date)
    panel=panel.sort_values(["series_id","date"]).reset_index(drop=True)
    group_index=panel.groupby("series_id",sort=False).indices
    excluded={x[0] for x in CASES}
    cases=[];controls=[];all_events=[]
    for case in CASES:
        code,name,asof,kind,label,key,description=case
        if label in DISCARD_LABELS:continue
        hist,origin=case_history(panel,group_index,case)
        if hist is None:
            cases.append(dict(code=code,analysis_date=asof,case_family=kind,
                              label=label,status=origin));continue
        s=canonical_row(hist,asof,kind,origin)
        s["case_id"]=code+"_"+asof;s["analysis_date"]=asof
        s["source_label_type"]=label;s["official_description"]=description
        s["official_source_url"]="https://contents.premium.naver.com/jusikdante/jusikdante1/contents/"+key
        s["exchange_code"]=str(hist.exchange.iloc[-1])
        cases.append(s)
        if s.get("status")!="REPLAY_READY":
            print("case",code,"unusable",s.get("status"),flush=True)
            continue
        comparisons=market_controls(panel,group_index,case,s,excluded,args.controls_per_case)
        for c in comparisons:
            c["case_id"]=s["case_id"];c["official_case_code"]=code
            controls.append(c)
        print("case",code,name,"asof",asof,"state",s.get("target_state"),
              "event20",s.get("candidate_event_last20"),
              "turnoverB",round(s["avg_turnover_prior20_krw"]/1e9,3),
              "controls",len(comparisons),flush=True)
    cdf=pd.DataFrame(cases);ref=pd.DataFrame(controls)
    cdf.to_csv(args.out/"official_asof_state_features.csv",index=False,encoding="utf-8-sig")
    ref.to_csv(args.out/"matched_unlabeled_market_controls.csv",index=False,encoding="utf-8-sig")
    summary=[]
    for family,g in cdf[cdf.status.eq("REPLAY_READY")].groupby("case_family"):
        q=ref[ref.case_family.eq(family)] if not ref.empty else pd.DataFrame()
        matched=q.target_state.eq(True).sum() if not q.empty else 0
        summary.append(dict(
            family=family,official_cases=int(len(g)),observed_state_matches=int(g.target_state.eq(True).sum()),
            observed_event_last20=int(g.candidate_event_last20.eq(True).sum()),
            unlabeled_control_stockdays=int(len(q)),
            matching_state_unlabeled_controls=int(matched),
            state_rate_in_official_context_pct=float(100*g.target_state.eq(True).mean()),
            state_rate_in_unlabeled_controls_pct=float(100*q.target_state.eq(True).mean()) if len(q) else np.nan,
        ))
    sm=pd.DataFrame(summary);sm.to_csv(args.out/"case_vs_market_prevalence.csv",index=False,encoding="utf-8-sig")
    gates=[]
    g=cdf[cdf.status.eq("REPLAY_READY")]
    for amt in (0,5e8,1e9,2e9,5e9):
        gates.append(dict(
            min_average_turnover_krw=int(amt),
            available_source_cases=int(len(g)),
            passing_cases=int((g.avg_turnover_prior20_krw>=amt).sum()),
            passing_pct=float(100*(g.avg_turnover_prior20_krw>=amt).mean()),
            case_state_match=int((g.target_state.eq(True)&g.avg_turnover_prior20_krw.ge(amt)).sum()),
            case_20d_event_match=int((g.candidate_event_last20.eq(True)&g.avg_turnover_prior20_krw.ge(amt)).sum()),
        ))
    pd.DataFrame(gates).to_csv(args.out/"liquidity_gate_sensitivity.csv",index=False,encoding="utf-8-sig")
    (args.out/"LIMITATIONS.txt").write_text(
        "This is a descriptive structural comparison, NOT validation of proprietary signals. "
        "Naver Premium articles are contextually positive setups, not dated actual paid "
        "scanner trigger labels. Same-date randomly selected matched symbols are UNLABELED, "
        "not true proprietary signal negatives. No F1/precision/recall can be estimated. "
        "All features are computed using candles only up to the corresponding article "
        "analysis date. Turnover gates and stable-base thresholds are independent "
        "research assumptions, not disclosed Dante formulas. Data from marcap is research "
        "adjusted and two cases use separately tagged Naver price histories. "
        "This code doesn't place stock orders or change live alerts.\n",
        encoding="utf-8")
    print("CASES",len(cdf),"CONTROL_ROWS",len(ref),flush=True)
    print("STATE TABLE\n"+sm.to_string(index=False),flush=True)
    print("LIQUIDITY\n"+pd.DataFrame(gates).to_string(index=False),flush=True)

if __name__=="__main__":
    main()
