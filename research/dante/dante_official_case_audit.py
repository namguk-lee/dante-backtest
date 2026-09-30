#!/usr/bin/env python3
"""Dated public-label comparison; reconstructed overlays are NOT visual verification.

No future bars enter a detector. Unmentioned proprietary signals are UNKNOWN.
Universe filtering is bypassed solely for labeled-case diagnostics; this does
not change the live scanner's 520-bar/turnover/EMA448 requirements.
"""
import argparse
import html
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import dante_signal_matrix_research as matrix


def plot_case(z,r,lab,path):
    t=z.tail(280).reset_index(drop=True)
    factor=float(t.iloc[-1].ac/t.iloc[-1].close)
    fig,(ax,av)=plt.subplots(2,1,figsize=(14,8),sharex=True,
        gridspec_kw={'height_ratios':[4,1]})
    for i,b in t.iterrows():
        color='#d34848' if b.ac>=b.ao else '#3976b7'
        ax.vlines(i,b.al/factor,b.ah/factor,color=color,lw=.65)
        bottom=min(b.ao,b.ac)/factor
        height=max(abs(b.ac-b.ao)/factor,float(b.ac/factor)*.0003)
        ax.add_patch(Rectangle((i-.32,bottom),.64,height,color=color))
    for n,col in [(112,'#6955b8'),(224,'#222222'),(448,'#cc8c16')]:
        ax.plot(t.index,t[f'ema{n}']/factor,label=f'EMA{n}',color=col,lw=1.4)
    ax.plot(t.index,t.bb35_upper2/factor,'--',color='#6790bc',lw=.8,label='BB35 proxy (unconfirmed)')
    markers=[('concrete_break_date','Break'),('concrete_retest_date','Retest'),
             ('bowl_peak_date','Bowl peak proxy'),('bowl_trough_date','Bowl trough proxy'),
             ('anchor_proxy_date','Volume candle proxy')]
    for key,label in markers:
        dt=r.get(key)
        if pd.notna(dt):
            ids=t.index[t.date.eq(pd.Timestamp(dt))]
            if len(ids):ax.axvline(ids[0],lw=.8,ls=':',label=f'{label}: {pd.Timestamp(dt):%Y-%m-%d}')
    if pd.notna(r.get('concrete_level')):
        ax.axhline(r['concrete_level'],ls=':',color='#39846b',label='Support proxy')
    ax.axvline(len(t)-1,color='#333333',ls='--',lw=.8)
    ax.set_title(f"{r['code']} | analysis {lab['analysis_date']:%Y-%m-%d} | "
                 f"market {t.iloc[-1].date:%Y-%m-%d} | {lab['outcome']} | RESEARCH ONLY")
    ax.text(.01,.98,r['signals'] or 'No current detector match',transform=ax.transAxes,
            va='top',fontsize=8)
    ax.legend(fontsize=7,ncol=3,loc='lower left'); ax.grid(alpha=.15)
    av.bar(t.index,t.volume,color='#8d9ead',width=.7)
    av.plot(t.index,t.v20,color='#d69b35',label='V20'); av.legend(fontsize=8)
    ticks=np.linspace(0,len(t)-1,8,dtype=int)
    av.set_xticks(ticks);av.set_xticklabels([f'{t.at[i,"date"]:%y-%m-%d}' for i in ticks],rotation=25)
    fig.tight_layout();fig.savefig(path,dpi=130);plt.close(fig)


def audit(panel,labels,out,baseline=None):
    out.mkdir(parents=True,exist_ok=True);(out/'charts').mkdir(exist_ok=True)
    rows=[];misses=[]
    for lab in labels.to_dict('records'):
        dt=pd.Timestamp(lab['analysis_date'])
        q=panel[(panel.name.astype(str).str.strip()==str(lab['stock']).strip()) & (panel.date<=dt)]
        if q.empty:
            misses.append({**lab,'reason':'no_history_before_analysis_date'});continue
        # Most recent listing episode at the analysis date, never longest history.
        sid=q.groupby('series_id').date.max().idxmax()
        # Once resolved, retain the episode's older names as well. Filtering
        # the entire history by today's name silently shortens renamed stocks.
        g=panel[panel.series_id.eq(sid) & (panel.date<=dt)].sort_values('date')
        market_dt=pd.Timestamp(g.iloc[-1].date)
        if (dt-market_dt).days>7:
            misses.append({**lab,'reason':'stale_history','market_date':market_dt,
                           'stale_calendar_days':(dt-market_dt).days});continue
        z=matrix.features(g)
        r=matrix.scan(z,0,apply_universe_filter=False)
        r.update({'stock':lab['stock'],'analysis_date':dt,'market_date':z.iloc[-1].date,
            'history_rows':len(z),'ema448_available':pd.notna(z.iloc[-1].ema448),
            'outcome_label':lab['outcome'],'stop_hit_label':lab['stop_hit'],
            'official_techniques':lab['techniques'],'official_context':lab['context_tags'],
            'official_vip_mentions':lab['vip_indicators'],'source_url':lab['source_url'],
            'visual_match_status':'NOT_REVIEWED','unmentioned_signal_status':'UNKNOWN',
            'universe_filter_bypassed':True,
            'data_source':g.iloc[-1].get('source','MARCAP_RESEARCH_ADJUSTED'),
            'stale_calendar_days':(dt-market_dt).days})
        if baseline:
            old=baseline.concrete_proxy(z)
            r['concrete_before']=old[0]
            r['concrete_before_break_date']=old[2]
            r['concrete_before_retest_date']=old[3]
        anchor=z.tail(20)
        anchor=anchor[(anchor.ret1>=.05)&(anchor.body>=.035)&(anchor.vr20>=2)]
        r['anchor_proxy_date']=anchor.iloc[-1].date if len(anchor) else pd.NaT
        # Mention-level diagnostics only; other detectors firing is not a false positive.
        bowl_named='밥그릇' in str(lab['techniques'])
        r['official_bowl_mentioned']=bowl_named
        r['bowl_detector_on_named_case']=bool(r['public_bowl3_structure_proxy']) if bowl_named else None
        assert z.date.max()<=dt
        code=str(r['code']); filename=f'{code}_{dt:%Y%m%d}.png'
        r['chart']='charts/'+filename
        plot_case(z,r,{**lab,'analysis_date':dt},out/r['chart'])
        rows.append(r)
    result=pd.DataFrame(rows)
    result.to_csv(out/'official_case_audit.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(misses).to_csv(out/'official_case_misses.csv',index=False,encoding='utf-8-sig')
    def esc(v):return html.escape(str(v))
    cards=[]
    for r in rows:
        cards.append(f"<article><h2>{esc(r['stock'])} · {r['analysis_date']:%Y-%m-%d}</h2>"
          f"<p>공식 기법: {esc(r['official_techniques'])}<br>공식 문맥: {esc(r['official_context'])}</p>"
          f"<p>우리 탐지: {esc(r['signals'] or '현재 탐지 없음')}<br>"
          f"기법 표기 {r['public_technique_raw_count']}개 → 회복 계열 {r['public_technique_count']}개 · "
          f"공구리 {esc(r.get('concrete_before','미계산'))} → {esc(r['research_concrete_proxy'])} · "
          f"데이터 {r['history_rows']}봉</p><img src='{r['chart']}' loading='lazy'>"
          f"<p><a href='{esc(r['source_url'])}'>공식 사례 원문</a> · 원본 이미지와 육안 대조 미완료</p></article>")
    summary={'labels':len(labels),'matched':len(rows),'misses':len(misses),
      'overlapping_labels':(int((result.public_technique_raw_count>result.public_technique_count).sum()) if len(result) else 0),
      'concrete_after':(int(result.research_concrete_proxy.sum()) if len(result) else 0),
      'bowl_named':(int(result.official_bowl_mentioned.sum()) if len(result) else 0),
      'bowl_named_detected':(int(result.loc[result.official_bowl_mentioned,'public_bowl3_structure_proxy'].sum()) if len(result) else 0),
      'ema448_unavailable':(int((~result.ema448_available).sum()) if len(result) else 0)}
    if baseline:summary['concrete_before']=(int(result.concrete_before.sum()) if len(result) else 0)
    import json
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    page="""<!doctype html><html lang='ko'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>단테 공식 사례 대조</title><style>body{font:16px/1.65 system-ui;margin:auto;max-width:1100px;padding:24px;color:#172435;background:#f4f6f9}article{background:white;padding:20px;margin:24px 0;border-radius:14px}img{width:100%;height:auto}h1{font-size:28px}h2{font-size:20px}a{color:#285bb3}</style><h1>단테 공식 사례 22개 · 날짜별 탐지 대조</h1><p>각 분석일까지의 데이터만 사용했습니다. 이 차트는 우리 계산을 표시한 재구성 차트입니다. 공식 이미지와 일치한다고 검증된 결과가 아닙니다. 공식 수익률은 성과 라벨이며 우리 전략의 수익률이 아닙니다. 수박·레인보우는 재현하지 않았고, BB35는 미확정 가설입니다.</p>"""
    (out/'index.html').write_text(page+'<pre>'+esc(summary)+'</pre>'+''.join(cards)+'</html>')
    print(summary)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--panel',type=Path,required=True)
    p.add_argument('--labels',type=Path,default=Path(__file__).with_name('dante_official_examples.csv'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--baseline',type=Path)
    a=p.parse_args()
    panel=pd.read_parquet(a.panel);panel['date']=pd.to_datetime(panel.date)
    labels=pd.read_csv(a.labels);labels['analysis_date']=pd.to_datetime(labels.analysis_date)
    baseline=None
    if a.baseline:
        spec=importlib.util.spec_from_file_location('matrix_baseline',a.baseline)
        baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)
    audit(panel,labels,a.out,baseline)


if __name__=='__main__':main()
