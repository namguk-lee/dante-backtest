#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replay independently documented Dante analysis examples, AS OF THEIR dates.

Critical epistemic boundary: a dated stock-analysis article is usually NOT a
paid-screener signal timestamp. Reports label whether a case is a claimed
screener member, a setup description, an earlier example, or a hypothetical.
Do not report precision/recall without the vendor's exact event output and
negative stock-days. Research numeric proxies are not paid proprietary code.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from dante_independent_indicator_suite import prepare, event_tests

ROOT="https://contents.premium.naver.com/jusikdante/jusikdante1/contents/"
# Printed analysis date may be much earlier than the later retrospective post.
CASES=[
 ("156100","엘앤케이바이오","2026-05-19","ODORI","SCREEN_MEMBER_DATE_UNDISCLOSED","260519170521976rz","오돌이 검색기 종목 언급; 당일 신호시점 미공개"),
 ("137400","피엔티","2025-04-23","ODORI","PRIOR_TREND_BREAK","250424173409705aq","오돌이 추세 돌파 후 흐름; 실제 돌파일 미공개"),
 ("238490","힘스","2025-07-17","ODORI","EARLIER_HISTORICAL","250717173549854kz","이전에 오돌이 자리에서 분석; 이번날 신호라고 하지 않음"),
 ("347860","알체라","2025-06-02","EMA112","SETTLED_AFTER_FIRST_CROSS","250602164532580fg","112일선 돌파 후 눌림; 신규 돌파일은 더 이전"),
 ("146320","비씨엔씨","2025-08-08","REVERSE_BASE","SETUP_RECAP","251002164439202kg","역매공파로 분석한 자리; 색변환 점선 원시값 미공개"),
 ("419050","삼기에너지솔루션즈","2025-07-21","REVERSE_BASE","SETUP_RECAP","250821161412291pr","60일선 안착/매집봉/작은 공구리"),
 ("465770","STX그린로지스","2026-03-09","REVERSE_BASE","SETUP_RECAP","260508173158169ph","역배열, 112일 안착, 매집봉, 파란점선"),
 ("084850","아이티엠반도체","2026-03-30","REVERSE_BASE","SETUP_RECAP","260409171022476in","기준봉 후 눌림과 역매공파"),
 ("004980","성신양회","2025-12-09","BOWL_BASE","SETUP_RECAP","260416172737910pf","밥그릇 1/2와 매집봉·공구리"),
 ("025550","한국선재","2026-04-20","ANCHOR","SETUP_RECAP","260515165352528il","기준봉·매집과 공격형 밥그릇 패턴"),
 ("366030","공구우먼","2025-11-03","BOWL_BASE","SETUP_RECAP","251112164213072mf","2번 밥그릇 끝 삼각수렴 돌파"),
 ("424760","벨로크","2025-09-08","REVERSE_BASE","SETUP_RECAP","251210171513294nm","역매공파 및 바닥 쓰리바닥"),
 ("199550","레이저옵텍","2025-07-16","ANCHOR","EXACT_DAY_CANDLE","250911170904298ym","7월16일 캔들이 변곡점·기준봉이라고 특정"),
 ("291810","핀텔","2025-07-09","ODORI","FUTURE_POSSIBILITY","250923163734334zx","오돌이 자리일 수도 있다고 설명; 정답 레이블 아님"),
 ("087600","픽셀플러스","2025-03-24","ODORI","FUTURE_POSSIBILITY","250324164703192zu","향후 하락 시 오돌이 자리를 다시 노린다는 가정"),
]
MAP={
 "ODORI":{"ODORI_EMA5_PROXY","ODORI_SMA5_PROXY","BOTTOM_ODORI_PROXY","REVERSE_ODORI_PROXY"},
 "EMA112":{"MA_HIT_112_PROXY"},
 "REVERSE_BASE":{"REVERSE_ACCUM_CONCRETE_PROXY","REVERSE_ODORI_PROXY","MA_HIT_112_PROXY"},
 "BOWL_BASE":{"REVERSE_ACCUM_CONCRETE_PROXY","MA_HIT_224_PROXY"},
 "ANCHOR":{"REVERSE_ACCUM_CONCRETE_PROXY","MA_HIT_112_PROXY","MA_HIT_224_PROXY"},
}

def asof_probe(g,case):
    code,name,anal,kind,label,key,desc=case
    z=prepare(g)
    asof=pd.Timestamp(anal)
    past=z.index[z.date.le(asof)]
    if not len(past):
        return dict(status="NO_HISTORY_BEFORE_CASE")
    idx=int(past[-1]);bar=z.iloc[idx]
    rec=dict(date=anal,observed_price_date=str(bar.date.date()),
       code=code,official_name=name,market_name=str(bar["name"]),
       source_publication_date=None,source_url=ROOT+key,
       category=kind,source_label_type=label,official_description=desc,
       episode=str(bar.series_id),history_bars=int(idx+1),
       raw_close=float(bar.close),
       avg_turnover_prior20=int(bar.preamount20) if pd.notna(bar.preamount20) else 0,
       meets_current_50e_gate=bool(pd.notna(bar.preamount20) and bar.preamount20>=5e9),
       year_to_analysis_date_gap=(asof-bar.date).days)
    if idx<560:
        rec.update(status="SHORT_EPISODE_LT560",proxy_fired_same_day=False,
                   proxy_fired_in_last20=False,matching_proxy_dates="",all_recent_proxies="")
        return rec
    last20={};all20={}
    for i in range(max(560,idx-20),idx+1):
        if not pd.notna(z.preamount20.iloc[i]):continue
        day=str(z.date.iloc[i].date())
        for model,reason,base in event_tests(z,i):
            all20.setdefault(model,[]).append(day)
            if model in MAP[kind]:
                last20.setdefault(model,[]).append(day)
    rec.update(
       status="REPLAY_READY",
       ema5=float(bar.ema5),sma5=float(bar.sma5),
       ema112=float(bar.ema112),ema224=float(bar.ema224),
       ema448=float(bar.ema448),
       ema5_delta_pct=float(100*(bar.ema5/bar.ema5.shift(1)-1)) if False else float(100*(bar.ema5/z.ema5.iloc[idx-1]-1)),
       ema5_upturn_today=bool(bar.ema5_upturn),
       sma5_upturn_today=bool(bar.sma5_upturn),
       close_above_ema112=bool(bar.ac>=bar.ema112),
       close_above_ema224=bool(bar.ac>=bar.ema224),
       long_reverse_112_224_448=bool(bar.ema112<bar.ema224<bar.ema448),
       lower_ma_reverse_recent60=bool(((z.ema112<z.ema224)&(z.ema224<z.ema448)).iloc[max(0,idx-59):idx+1].any()),
       volume_ratio_pre20=float(bar.volratio),
       closepos_pct=float(100*bar.closepos) if pd.notna(bar.closepos) else np.nan,
       anchor_candle_today=bool(bar.volratio>=1.8 and bar.body>=.025 and bar.closepos>=.65),
       recent90_anchor_count=int(((z.volratio>=1.8)&(z.body>=.025)&(z.closepos>=.65)).iloc[max(0,idx-89):idx+1].sum()),
       close_to_ema224_pct=float(100*(bar.ac/bar.ema224-1)),
       proxy_fired_same_day=any(str(bar.date.date()) in ds for ds in last20.values()),
       proxy_fired_in_last20=bool(last20),
       matching_proxy_dates=" | ".join(m+":"+",".join(d) for m,d in last20.items()),
       all_recent_proxies=" | ".join(m+":"+",".join(ds) for m,ds in all20.items()),
    )
    return rec

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--ko",type=Path,required=True)
    p.add_argument("--kq",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    from dante_classic_scanner import load
    args.out.mkdir(parents=True,exist_ok=True)
    src_codes={c[0] for c in CASES}
    panel=pd.concat([load(args.ko,"KO"),load(args.kq,"KQ")],ignore_index=True)
    panel=panel[panel.code.isin(src_codes)].copy()
    output=[]
    for case in CASES:
        code,name,anal,kind,label,key,desc=case
        q=panel[panel.code.eq(code)&panel.date.le(pd.Timestamp(anal))]
        if q.empty:
            output.append(dict(date=anal,code=code,official_name=name,
                               category=kind,source_label_type=label,
                               source_url=ROOT+key,status="NO_MARKET_EPISODE"));continue
        latest=q.sort_values("date").iloc[-1]
        g=panel[panel.series_id.eq(latest.series_id)].copy()
        r=asof_probe(g,case)
        output.append(r)
        print(code,name,anal,r["status"],"liquid",r.get("meets_current_50e_gate"),
              "proxy20",r.get("proxy_fired_in_last20"),flush=True)
    audit=pd.DataFrame(output)
    audit.to_csv(args.out/"official_examples_asof_replay.csv",index=False,encoding="utf-8-sig")
    official_valid=audit[~audit.source_label_type.isin(["FUTURE_POSSIBILITY","EARLIER_HISTORICAL"])]
    a=official_valid[official_valid.status.eq("REPLAY_READY")]
    s=dict(
       official_cited_cases=len(audit),
       strict_asof_comparable_cases=len(official_valid),
       with_sufficient_history=len(a),
       with_proxy_in_20trading_days=int(a.proxy_fired_in_last20.sum()) if len(a) else 0,
       with_proxy_same_day=int(a.proxy_fired_same_day.sum()) if len(a) else 0,
       liquidity_gate_over_50e=int(a.meets_current_50e_gate.sum()) if len(a) else 0,
       missing_history=len(official_valid)-len(a),
       original_paid_signal_true_positives="NOT_OBSERVABLE",
       original_paid_signal_true_negatives="NOT_OBSERVABLE",
       original_indicator_precision_recall="NOT_IDENTIFIABLE"
    )
    pd.DataFrame([s]).to_csv(args.out/"audit_summary.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(CASES,columns=["code","name","analysis_date","family","label_class","source_id","official_context"]).to_csv(
        args.out/"official_grounded_cases.csv",index=False,encoding="utf-8-sig")
    (args.out/"README.txt").write_text(
       "Independent Dante-inspired replay against public officially dated articles. "
       "Analysis date is not exact paid screener firing date unless separately documented. "
       "20 preceding TRADING days proxy matching is a proximity diagnostic, NOT a vendor "
       "precision/recall score. The 50e KRW liquidity gate is a research choice, not "
       "a disclosed Dante gate. Missing signals must not be counted as paid-formula mismatch "
       "until as-of source data, history episodes, and event timing are validated. "
       "No trades. Every event uses only bars known as of the analysis date.\n",encoding="utf-8")
    print("SUMMARY",s,flush=True)

if __name__=="__main__":main()
