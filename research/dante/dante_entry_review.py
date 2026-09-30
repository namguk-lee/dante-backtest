#!/usr/bin/env python3
"""Dated review of existing classic anchor rules; never a validated buy signal."""
import argparse,base64,html
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dante_classic_scanner import classify_one,find_concrete
from dante_signal_matrix_research import features,scan
from collect_marcap_full_market import last_closed_date


def review(g,end,min_turnover=5e9):
    g=g[pd.to_datetime(g.date)<=pd.Timestamp(end)].copy()
    if g.empty:raise ValueError('no closed history before cutoff')
    if pd.Timestamp(end)>last_closed_date():raise ValueError('review cutoff includes an incomplete session')
    if g.date.max()!=pd.Timestamp(end):raise ValueError('review history is stale')
    r,z=classify_one(g,min_turnover)
    broad=scan(features(g),min_turnover)
    if r is None:
        raise ValueError(f'{g.code.iloc[-1]}: no classic anchor to review; no buy signal')
    ai=int(r['_idx_anchor']);anchor=float(z.at[ai,'ao'])
    breaches=z.iloc[ai+1:][z.iloc[ai+1:].ac<anchor]
    r['first_anchor_break_date']=breaches.date.iloc[0] if not breaches.empty else pd.NaT
    r['first_anchor_break_close']=float(breaches.close.iloc[0]) if not breaches.empty else np.nan
    concrete=find_concrete(z,ai)
    r['concrete_break_date']=pd.NaT;r['first_concrete_breakdown_date']=pd.NaT
    if concrete and concrete.get('concrete_breakout'):
        bi=int(concrete['concrete_breakout_i']);r['concrete_break_date']=z.at[bi,'date']
        broken=z.iloc[bi:][z.iloc[bi:].ac<float(concrete['resistance_adj'])]
        if not broken.empty:r['first_concrete_breakdown_date']=broken.date.iloc[0]
    r['analysis_date']=pd.Timestamp(end)
    r['broad_signals']=broad['signals'] if broad else ''
    r['broad_concrete_proxy']=bool(broad and broad['research_concrete_proxy'])
    r['decision']='WAIT_NEW_ANCHOR' if not r['anchor_alive_strict'] else 'WATCH_UNVALIDATED'
    r['entry_validation']='NOT_VALIDATED';r['buy_signal']=False
    return r,z


def plot(z,r,path):
    t=z.tail(65).reset_index(drop=True);factor=float(t.iloc[-1].ac/t.iloc[-1].close)
    fig,(ax,av)=plt.subplots(2,1,figsize=(13,8),sharex=True,gridspec_kw={'height_ratios':[4,1]})
    for i,b in t.iterrows():
        color='#d64a4a' if b.ac>=b.ao else '#3376b8'
        ax.vlines(i,b.al/factor,b.ah/factor,color=color,lw=.8)
        ax.add_patch(Rectangle((i-.3,min(b.ao,b.ac)/factor),.6,max(abs(b.ac-b.ao)/factor,b.ac/factor*.0004),color=color))
    for n,col in [(112,'#6b55bd'),(224,'#303b49'),(448,'#c68b21')]:
        ax.plot(t.index,t[f'ema{n}']/factor,label=f'EMA{n}',color=col,lw=1.3)
    for key,label,color in [('anchor_date','Anchor','#3a8870'),('first_anchor_break_date','First anchor breach','#cf4242'),('concrete_break_date','Prior-high breakout','#3678ab'),('first_concrete_breakdown_date','Support breach','#b95a4b')]:
        dt=r.get(key)
        if pd.notna(dt):
            ids=t.index[t.date.eq(pd.Timestamp(dt))]
            if len(ids):ax.axvline(ids[0],color=color,ls=':',lw=1,label=f'{label}: {pd.Timestamp(dt):%Y-%m-%d}')
    if pd.notna(r.get('anchor_open')):ax.axhline(r['anchor_open'],color='#3a8870',ls='--',label=f'Anchor open {r["anchor_open"]:,.0f}')
    if pd.notna(r.get('concrete_resistance')):ax.axhline(r['concrete_resistance'],color='#3678ab',ls='--',label=f'Prior-high level {r["concrete_resistance"]:,.0f}')
    ax.set_title(f'{r["code"]} | {r["analysis_date"]:%Y-%m-%d} | {r["decision"]} | NO VALIDATED BUY')
    ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.15)
    av.bar(t.index,t.volume,color='#91a6bd');av.plot(t.index,t.v20,label='Volume MA20',color='#cc932e');av.legend(fontsize=8)
    ids=np.linspace(0,len(t)-1,8,dtype=int);av.set_xticks(ids);av.set_xticklabels([f'{t.at[i,"date"]:%m-%d}' for i in ids]);fig.tight_layout();fig.savefig(path,dpi=135);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--panel',type=Path,required=True);p.add_argument('--end',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--codes',nargs='+',required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    panel=pd.read_parquet(a.panel);panel['date']=pd.to_datetime(panel.date);panel['code']=panel.code.astype(str).str.zfill(6)
    rows=[];cards=[]
    for c in a.codes:
        g=panel[panel.code.eq(c)];sid=g.groupby('series_id').date.max().idxmax();g=g[g.series_id.eq(sid)]
        r,z=review(g,a.end);rows.append(r);path=a.out/f'{c}_entry_review.png';plot(z,r,path)
        img=base64.b64encode(path.read_bytes()).decode();date=lambda v:f'{pd.Timestamp(v):%Y-%m-%d}' if pd.notna(v) else '없음'
        cards.append(f'''<section><h2>{html.escape(r['name'])} ({c}) · 매수 보류</h2><table><tr><th>기준 종가</th><td>{r['current_close']:,.0f}원</td><th>엄격한 앵커 유지</th><td>{r['anchor_alive_strict']}</td></tr><tr><th>강한 상승 앵커</th><td>{date(r['anchor_date'])}</td><th>앵커 시가</th><td>{r['anchor_open']:,.0f}원</td></tr><tr><th>첫 앵커 이탈</th><td>{date(r['first_anchor_break_date'])}</td><th>이탈일 종가</th><td>{r['first_anchor_break_close']:,.0f}원</td></tr><tr><th>이전 고점 돌파</th><td>{date(r['concrete_break_date'])}</td><th>돌파선</th><td>{r['concrete_resistance']:,.0f}원</td></tr><tr><th>첫 돌파선 이탈</th><td>{date(r['first_concrete_breakdown_date'])}</td><th>엄격한 돌파선 유지</th><td>{r['concrete_hold_strict']}</td></tr></table><p>EMA112 {r['ema112']:,.0f}원 · EMA224 {r['ema224']:,.0f}원 · EMA448 {r['ema448']:,.0f}원</p><p>넓은 검색: {html.escape(r['broad_signals'])}. 넓은 콘크리트 근사 표시: {r['broad_concrete_proxy']}. 이것은 앵커가 살아 있다는 뜻이 아니다.</p><img src="data:image/png;base64,{img}" alt="{c} 65봉 가격·거래량과 앵커 이탈 날짜"><p><a href="https://kind.krx.co.kr/common/stockprices.do?method=searchStockPricesMain&amp;isurCd={c[:5]}">한국거래소 시세</a></p></section>''')
    pd.DataFrame(rows).to_csv(a.out/'entry_review.csv',index=False,encoding='utf-8-sig')
    body='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>단테 진입 정밀 검증</title><style>body{max-width:1100px;margin:32px auto;padding:0 20px;font:16px/1.7 sans-serif;color:#213047}h1{font-size:27px}section{margin-top:32px;padding-top:15px;border-top:1px solid #ccd6e2}img{width:100%}table{width:100%;border-collapse:collapse}td,th{padding:9px;border-bottom:1px solid #dee5ed;text-align:left}th{background:#f0f4f8}.note{padding:18px;background:#fff2df;border-radius:8px}a{color:#145aa1}@media(max-width:600px){table{font-size:12px}td,th{padding:5px}}</style><h1>단테 진입 정밀 검증 · 2026-09-29 확정봉</h1><div class="note"><b>결론: 우리기술투자·헥토파이낸셜 모두 매수 보류, 새 앵커 대기.</b><br>앞선 넓은 검색 추천은 엄격한 앵커 유지 검증을 통과하지 못했다. 최근 거래량 감소나 이평선 근접만으로 매수하지 않는다. 오늘 9/30 진행 중인 봉은 평가에서 제외했다.</div><p>기존 classic 연구 모듈의 규칙을 그대로 비교했다. 앵커 후보는 몸통 상승률 ≥3.5%, 거래량/20일평균 ≥1.8, 봉 종가 위치 ≥60%, EMA112 위 또는 20봉 고점 돌파, 20일 평균 거래대금 ≥50억원으로 정의된 근사다. 이후 종가가 앵커 시가 아래로 한번이라도 내려오면 엄격한 앵커 유지에 실패한다. 현재 앵커를 재사용하지 않고 새 강한 상승봉과 후속 지지를 기다리는 관찰 상태다. 이것은 단테 공식·독점 산식을 재현하거나 수익성이 입증된 진입 규칙이 아니다.</p>'''+''.join(cards)+'''<section><h2>데이터 확인과 한계</h2><p>과거 2022~2026: marcap KRX 참고 일봉, 수정된 KOSDAQ GLOBAL 분류, 연구용 기업행동 보정. 9/29 두 종목: 매일경제 정규장 OHLCV를 사용하고 한국거래소 KIND의 9/30 전일가로 종가를 독립 확인했다. KIND는 종가만 교차 검증하며 과거 OHLCV 전부를 검증하지 않는다. 거래대금은 공개 화면의 백만원 단위 반올림값이다. 차트는 보정 가격을 마지막 봉의 원화 단위로 환산한 가격이다.</p><p>NAVER 일반·신규 일봉 및 거래소 지정 차트 응답은 KIND 전일가와 불일치했다. 원인은 확정하지 않았으며, 단순히 NXT 차이라고 단정하지 않는다. 자동 보충을 차단하고 검토 기록을 가진 KRX 일봉만 입력하도록 수정했다. 최신 전시장 자동 수집 경로는 아직 미해결이며, 이 보고서는 두 종목 검증으로 제한된다. 동일 규칙의 전시장 수익성 백테스트는 이번 작업에 포함하지 않았다.</p><p><a href="https://stock.mk.co.kr/price/home/KR7041190000">우리기술투자 시세 출처</a> · <a href="https://stock.mk.co.kr/price/home/KR7234340008">헥토파이낸셜 시세 출처</a></p></section></html>'''
    (a.out/'dante-entry-review-20260929.html').write_text(body)
    print(pd.DataFrame(rows)[['code','name','anchor_date','anchor_open','first_anchor_break_date','first_anchor_break_close','concrete_break_date','first_concrete_breakdown_date','decision']].to_string(index=False))

if __name__=='__main__':main()
