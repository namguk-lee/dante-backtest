#!/usr/bin/env python3
"""Dated, incomplete-coverage research review; never emits a buy signal."""
import argparse, base64, html, json
from pathlib import Path
import numpy as np
import pandas as pd
from dante_classic_scanner import classify_one, find_concrete, local_swing_highs
from dante_entry_review import plot
from collect_marcap_full_market import last_closed_date


def validate_recent_coverage(recent,ledger):
    expected=set(ledger.loc[ledger.status.eq('OK_RESEARCH_ONLY'),'code'])
    if set(recent.code)!=expected:raise ValueError('recent bars and collection ledger coverage disagree')
    if recent.duplicated(['code','date']).any():raise ValueError('duplicate recent bar')
    for code,g in recent.groupby('code'):
        if set(pd.to_datetime(g.date))!={pd.Timestamp('2026-09-29'),pd.Timestamp('2026-09-30')}:
            raise ValueError(f'{code}: incomplete recent dates')
    if pd.Timestamp('2026-09-30')>last_closed_date():raise ValueError('incomplete review session')


def append_research(g, recent):
    g=g.sort_values('date').copy()
    tail=g.iloc[-1];factor=float(tail.adjusted_close/tail.close)
    rows=[]
    for row in recent.sort_values('date').to_dict('records'):
        row.update(series_id=tail.series_id,adjusted_close=row['close']*factor,
                   reported_change_pct=np.nan,adjustment_bridge=False)
        rows.append(row)
    return pd.concat([g,pd.DataFrame(rows)],ignore_index=True)


def qualify(r,z,g):
    """Additional conservative research gates, not claimed official rules."""
    reasons=[]
    if r['tier'] not in ('A','B'):reasons.append('classic structure incomplete')
    if not r['anchor_alive_strict']:reasons.append('anchor breached')
    if not r['concrete_hold_strict']:reasons.append('prior-high support not strictly held')
    if pd.Timestamp(r['anchor_date'])>=pd.Timestamp('2026-09-29'):reasons.append('anchor volume unverified')
    conservative=g.copy();mask=conservative.date>=pd.Timestamp('2026-09-29')
    conservative.loc[mask,['volume','amount']]=0
    r0,_=classify_one(conservative,5e9)
    if r0 is None or any(r[k]!=r0[k] for k in ('anchor_date','anchor_alive_strict','concrete_hold_strict','tier')):
        reasons.append('recent volume sensitivity')
    if conservative.amount.tail(20).sum()/20<5e9:reasons.append('conservative liquidity below 5bn won/day')
    concrete=find_concrete(z,int(r['_idx_anchor']))
    retest=pd.NaT
    if concrete and concrete.get('concrete_breakout'):
        level=float(concrete['resistance_adj']);bi=int(concrete['concrete_breakout_i'])
        later=z.iloc[bi+1:]
        hits=later[(later.al>=level*.97)&(later.al<=level*1.03)&(later.ac>=level)]
        if len(hits):retest=hits.date.iloc[0]
    if pd.isna(retest):reasons.append('no separate-bar support retest')
    stop=max(r['anchor_open'],r['concrete_resistance']) if pd.notna(r['concrete_resistance']) else r['anchor_open']
    risk=(r['current_close']-stop)/r['current_close']*100 if stop>0 else np.nan
    # Include even nearby overhead pivots omitted by the classic >3% rule.
    factor=float(z.iloc[-1].ac/z.iloc[-1].close)
    pivots=local_swing_highs(z,max(448,len(z)-261),len(z)-5,wing=4)
    overhead=[float(z.at[i,'ah']/factor) for i in pivots if z.at[i,'ah']/factor>r['current_close']]
    targets=overhead+([r['target_price']] if pd.notna(r['target_price']) else [])
    target=min(targets) if targets else np.nan
    reward=(target/r['current_close']-1)*100 if pd.notna(target) else np.nan
    rr=reward/risk if pd.notna(reward) and risk>0 else np.nan
    if not (0<risk<=10):reasons.append('support distance outside 0-10%')
    if not (pd.notna(reward) and reward>=5 and pd.notna(rr) and rr>=2):reasons.append('insufficient overhead room / reward-risk')
    r.update(research_qualified=not reasons,rejection_reasons='; '.join(reasons),retest_date=retest,
             review_support=stop,review_target=target,review_risk_pct=risk,review_reward_pct=reward,review_rr=rr,
             buy_signal=False,entry_validation='NOT_VALIDATED')
    return r


def main():
    p=argparse.ArgumentParser();p.add_argument('--panel',type=Path,required=True);p.add_argument('--recent',type=Path,required=True);p.add_argument('--ledger',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    panel=pd.read_parquet(a.panel);recent=pd.read_csv(a.recent,dtype={'code':str});recent['date']=pd.to_datetime(recent.date)
    ledger=pd.read_csv(a.ledger,dtype={'code':str})
    validate_recent_coverage(recent,ledger)
    active=panel[panel.date.eq(panel.date.max())].drop_duplicates('code')
    recent_map={c:q for c,q in recent.groupby('code')}
    histories=panel.groupby('series_id',sort=False).indices
    rows=[];charts=[];eligible=0;complete_history=0;classified=0;liquidity_excluded=0
    for _,meta in active.iterrows():
        g=panel.iloc[histories[meta.series_id]]
        if len(g)>=520:complete_history+=1
        if meta.code not in recent_map or len(g)<520:continue
        eligible+=1;g=append_research(g,recent_map[meta.code])
        # This gate is required by qualify(), so reject it before costly chart
        # classification. Keep the two earlier watch names for explicit review.
        if g.amount.iloc[-20:-2].sum()/20<5e9:
            liquidity_excluded+=1
            if meta.code not in ('041190','234340'):continue
        classified+=1;r,z=classify_one(g,5e9)
        if r is None:continue
        r=qualify(r,z,g);rows.append(r)
        if meta.code in ('041190','234340'):
            r['analysis_date']=pd.Timestamp('2026-09-30');r['decision']='WAIT_NEW_ANCHOR' if not r['anchor_alive_strict'] else 'WATCH_UNVALIDATED'
            breach=z.iloc[int(r['_idx_anchor'])+1:];breach=breach[breach.ac<z.at[int(r['_idx_anchor']),'ao']]
            r['first_anchor_break_date']=breach.date.iloc[0] if len(breach) else pd.NaT
            img=a.out/f'{meta.code}_review.png';plot(z,r,img)
            charts.append((r,base64.b64encode(img.read_bytes()).decode()))
        if eligible%200==0:print(f'scanned {eligible}, structures {len(rows)}',flush=True)
    q=pd.DataFrame(rows);q.to_csv(a.out/'scan_results.csv',index=False,encoding='utf-8-sig')
    qualified=q[q.research_qualified].sort_values('score',ascending=False) if len(q) else q
    qualified.to_csv(a.out/'qualified_research_watch.csv',index=False,encoding='utf-8-sig')
    summary=dict(analysis_date='2026-09-30',reference_active=len(active),reference_history_520plus=complete_history,
                 accepted_recent_codes=recent.code.nunique(),scanned_520plus_codes=eligible,
                 conservative_liquidity_excluded=liquidity_excluded,classic_classified_codes=classified,
                 structures=len(rows),strict_anchor=int(q.anchor_alive_strict.sum()) if len(q) else 0,
                 strict_anchor_and_support=int((q.anchor_alive_strict&q.concrete_hold_strict).sum()) if len(q) else 0,
                 research_qualified=len(qualified),validated_buy_signals=0,verified_full_market=False)
    (a.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    esc=lambda x:html.escape(str(x))
    labels={'analysis_date':'기준일','reference_active':'9/28 참고 자료의 상장 종목','reference_history_520plus':'520봉 이상 참고 이력',
            'accepted_recent_codes':'최근 가격 대조 통과','scanned_520plus_codes':'최근 가격 확인 · 520봉 이상 검사',
            'conservative_liquidity_excluded':'보수적 평균 거래대금 50억원 미달','classic_classified_codes':'차트 구조 정밀 판정 (기존 2종목 포함)',
            'structures':'앵커 포함 구조 검색','strict_anchor':'검색 구조 중 앵커 유지','strict_anchor_and_support':'검색 구조 중 앵커·돌파선 모두 유지',
            'research_qualified':'추가 조건 통과 관찰 후보','validated_buy_signals':'검증된 매수 신호','verified_full_market':'전시장 정규장 OHLCV 검증 완료'}
    table=''.join(f'<tr><td>{esc(labels[k])}</td><td>{"미완료" if k=="verified_full_market" else v}</td></tr>' for k,v in summary.items())
    translations={'classic structure incomplete':'classic 구조 조건 미충족','anchor breached':'앵커 시가 아래 종가 이탈',
                  'prior-high support not strictly held':'돌파선 지지 조건 미충족','anchor volume unverified':'앵커 거래량 미검증',
                  'recent volume sensitivity':'미확인 거래량에 따라 판정 변동','conservative liquidity below 5bn won/day':'보수적 평균 거래대금 50억원 미달',
                  'no separate-bar support retest':'돌파 이후 별도 봉의 지지 재확인 없음','support distance outside 0-10%':'지지선 거리 조건 미충족',
                  'insufficient overhead room / reward-risk':'상승 여유 또는 손익비 조건 미충족'}
    cards=''
    for r,img in charts:
        reasons=' · '.join(translations.get(reason,reason) for reason in r['rejection_reasons'].split('; '))
        cards+=f'<section><h2>{esc(r["name"])} ({r["code"]})</h2><p>9/30 종가 {r["current_close"]:,.0f}원 · 앵커 시가 {r["anchor_open"]:,.0f}원 · 앵커 유지 {"통과" if r["anchor_alive_strict"] else "실패"} · 돌파선 유지 {"통과" if r["concrete_hold_strict"] else "실패"}</p><p>제외 이유: {esc(reasons)}</p><img src="data:image/png;base64,{img}"></section>'
    watch=qualified[['code','name','current_close','anchor_date','review_support','review_target','review_rr']].head(3).to_html(index=False) if len(qualified) else '<p>모든 추가 조건을 통과한 관찰 후보가 없습니다.</p>'
    errors=ledger[ledger.status.eq('REJECTED')].error.value_counts().head(8).to_frame('count').to_html()
    report=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>단테 9/30 재점검</title><style>body{{max-width:1100px;margin:35px auto;padding:0 20px;font:16px/1.65 sans-serif;color:#24334a}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;border-bottom:1px solid #ccd6e2;text-align:left}}img{{width:100%}}section{{margin-top:30px}}.note{{padding:18px;background:#fff0d9}}</style><h1>단테 매매법 재점검 · 2026-09-30</h1><p class="note"><b>검증된 매수 신호 0개. 전시장 검증 완료가 아닙니다.</b> 최근 시세가 교차 확인된 종목만 연구용으로 재검사했습니다. 관찰 후보는 {len(qualified)}개이며 수익성이 입증된 추천주가 아닙니다.</p><h2>조회 범위와 결과</h2><table>{table}</table><h2>조건부 관찰</h2>{watch}<h2>판정 기준</h2><p>기존 classic 근사 규칙에서 A/B 구조, 앵커 시가 아래 종가 이탈 0회, 돌파선 아래 종가 이탈 0회가 필요합니다. 추가로 돌파 다음 봉 이후 ±3% 지지 재확인, 지지선까지 거리 0~10%, 가까운 과거 고점을 포함한 상승 여유 ≥5%, 여유/지지 거리 ≥2를 적용했습니다. 이 수치는 새 연구 필터이며 공식 단테 산식이 아닙니다. 9/29 이후 앵커는 거래량 미검증으로 제외하고, 최근 이틀의 거래량·거래대금을 0으로 둬도 판정이 유지되는지 검사했습니다. 20일 평균 거래대금 50억원은 최근 이틀을 0으로 둔 보수적 값으로도 넘어야 합니다.</p>{cards}<h2>데이터 한계</h2><p>9/28까지 FinanceData/marcap 참고 일봉과 연구용 기업행동 보정입니다. 9/30 KIND OHLC와 Yahoo OHLC를 대조했고 9/29 Yahoo 종가를 KIND 전일가로 확인했습니다. 9/29 시가·고가·저가 및 거래량은 전부 독립 확인되지 않았습니다. 9/29 거래대금은 저가×vendor 거래량 추정치입니다. KIND 거래량도 표시 마감 시각이 고정된 채 장 종료 후 변했으므로 정규장 전용으로 보증하지 않습니다. 출처 일치와 숫자 무결성 검사를 통과해도 완전한 정규장 OHLCV가 검증된 것은 아닙니다. 확인되지 않은 종목을 매수 후보로 취급하지 않습니다.</p><h2>조회 제외 원인</h2>{errors}<h2>다음 실행</h2><p>과거 일봉의 정규장 거래량·거래대금을 제공하는 검증 가능한 KRX 데이터 경로를 확보하고, 누락 종목까지 같은 기준으로 다시 검사해야 합니다. 그 다음 날짜를 분리한 백테스트와 거래비용을 포함한 성과 검증이 필요합니다. 현재 결과만으로 주문을 실행하지 않았습니다.</p><p>원본별 출처와 실패 이유는 collection_ledger.csv, 전체 판정은 scan_results.csv에 기록했습니다.</p></html>'''
    (a.out/'dante-review-20260930.html').write_text(report)
    print(json.dumps(summary,ensure_ascii=False));print(qualified[['code','name','score']].to_string(index=False) if len(qualified) else 'qualified=0')


if __name__=='__main__':main()
