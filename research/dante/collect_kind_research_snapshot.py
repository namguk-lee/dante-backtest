#!/usr/bin/env python3
"""Research-only recent bars. Never qualifies as a verified full-history overlay.

KIND supplies current KRX OHLC; Yahoo independently checks OHLC and supplies
the missing prior day's research bar. KIND volume changes after the displayed
closing timestamp, so neither source's volume is certified regular-session-only.
Prior-day amount is an estimate. All output remains research-only.
"""
import argparse, json, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup
from collect_marcap_full_market import last_closed_date


def parse_kind(text, code, end):
    s = BeautifulSoup(text, 'html.parser')
    identity = s.select_one('#repIsuSrtCd')
    if identity is None or identity.get('value') != 'A' + code:
        raise ValueError('KIND issuer/security mismatch')
    stamp = re.search(r'(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})\s*기준', s.get_text(' ', strip=True))
    if not stamp or stamp[1] != end or stamp[2] < '15:30:00':
        raise ValueError('KIND snapshot is stale or intraday')
    fields = {}
    for tr in s.select('table.detail tr'):
        cells = tr.find_all(['th', 'td'])
        for i in range(0, len(cells)-1, 2):
            label = cells[i].get_text(' ', strip=True).replace(' ', '')
            value = cells[i+1].get_text(' ', strip=True).replace(',', '')
            fields[label] = float(re.search(r'-?\d+(?:\.\d+)?', value)[0])
    def get(label):
        return next(v for k,v in fields.items() if k.startswith(label))
    row = dict(date=end, code=code, open=get('시가'), high=get('고가'), low=get('저가'),
               close=get('현재가'), volume=get('거래량'), amount=get('거래대금'),
               previous_close=get('전일가'), snapshot_time=stamp[2])
    validate_bar(row)
    return row


def validate_bar(row):
    if min(row[k] for k in ('open','high','low','close','volume','amount')) <= 0:
        raise ValueError('untraded/nonpositive bar')
    if row['high'] < max(row['open'],row['close'],row['low']) or row['low'] > min(row['open'],row['close'],row['high']):
        raise ValueError('invalid OHLC envelope')


def collect_one(meta, end, cache):
    code, exchange, name = meta
    path = cache / f'{code}_{end}.json'
    if path.exists():
        result=json.loads(path.read_text())
        result['cache_write_time_utc']=pd.Timestamp(path.stat().st_mtime,unit='s',tz='UTC').isoformat()
        return result
    source = f'https://kind.krx.co.kr/common/stockprices.do?method=searchStockPricesMain&isurCd={code[:5]}'
    yahoo_symbol = f'{code}.{"KS" if exchange == "KO" else "KQ"}'
    verify = f'https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}?range=5d&interval=1d'
    result = dict(code=code, name=name, exchange=exchange, source_url=source, verification_url=verify)
    try:
        headers={'User-Agent':'Mozilla/5.0'}
        response=requests.get(source,headers=headers,timeout=25);response.raise_for_status()
        response.encoding='utf-8'
        current=parse_kind(response.text,code,end)
        response=requests.get(verify,headers=headers,timeout=25);response.raise_for_status()
        chart=response.json()['chart']['result'][0]
        if chart['meta']['symbol'] != yahoo_symbol or chart['meta']['currency'] != 'KRW':
            raise ValueError('Yahoo identity/currency mismatch')
        quotes=chart['indicators']['quote'][0]
        bars={}
        for i,ts in enumerate(chart['timestamp']):
            date=pd.Timestamp(ts,unit='s',tz='UTC').tz_convert('Asia/Seoul').strftime('%Y-%m-%d')
            bars[date]={k:quotes[k][i] for k in ('open','high','low','close','volume')}
        if end not in bars or any(bars[end][k] != current[k] for k in ('open','high','low','close')):
            raise ValueError('current OHLC independent check failed')
        dates=sorted(d for d in bars if '2026-09-28' < d < end)
        if dates != ['2026-09-29'] or bars[dates[-1]]['close'] != current['previous_close']:
            raise ValueError('missing gap or prior close mismatch')
        prior=dict(bars[dates[-1]], date=dates[-1], code=code)
        # Conservative turnover estimate, not a verified traded amount.
        prior['amount']=prior['low']*prior['volume']
        validate_bar(prior)
        prior.update(verification_status='RESEARCH_GAP_VOLUME_UNVERIFIED',price_venue='OHLC_CROSSCHECK_ONLY',amount_precision='low * vendor volume estimate')
        current.update(verification_status='CURRENT_OHLC_CROSSCHECK_VOLUME_UNVERIFIED',price_venue='KRX',amount_precision='KIND won; session scope unverified')
        for row in (prior,current):
            row.update(exchange=exchange,name=name,source_url=source,verification_url=verify,bar_status='OHLC_CLOSED_VOLUME_SCOPE_UNVERIFIED',volume_scope='regular-session-only NOT VERIFIED')
        result.update(status='OK_RESEARCH_ONLY',rows=[prior,current],yahoo_current_volume=bars[end]['volume'])
    except Exception as exc:
        result.update(status='REJECTED',error=f'{type(exc).__name__}: {exc}')
    path.write_text(json.dumps(result,ensure_ascii=False))
    result['cache_write_time_utc']=pd.Timestamp(path.stat().st_mtime,unit='s',tz='UTC').isoformat()
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--panel',type=Path,required=True);p.add_argument('--end',required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=12);a=p.parse_args()
    if pd.Timestamp(a.end)>last_closed_date():raise ValueError('incomplete requested session')
    panel=pd.read_parquet(a.panel);latest=panel.date.max()
    if latest!=pd.Timestamp('2026-09-28') or a.end!='2026-09-30':raise ValueError('collector currently supports audited Sep28->Sep30 gap only')
    active=panel[panel.date.eq(latest)].drop_duplicates('code')
    metas=list(active[['code','exchange','name']].itertuples(index=False,name=None))
    a.cache.mkdir(parents=True,exist_ok=True);a.out.mkdir(parents=True,exist_ok=True)
    results=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(collect_one,m,a.end,a.cache) for m in metas]
        for f in as_completed(futures):
            result=f.result()
            for row in result.get('rows',[]):row['cache_write_time_utc']=result['cache_write_time_utc']
            results.append(result)
            if len(results)%100==0:print(f'collected {len(results)}/{len(metas)}, accepted {sum(r["status"]=="OK_RESEARCH_ONLY" for r in results)}',flush=True)
    ledger_path=a.out/'collection_ledger.csv'
    pd.DataFrame([{k:v for k,v in r.items() if k!='rows'} for r in results]).to_csv(ledger_path.with_suffix('.csv.tmp'),index=False)
    rows=[row for r in results if r['status']=='OK_RESEARCH_ONLY' for row in r['rows']]
    bars_path=a.out/'research_recent_bars.csv'
    pd.DataFrame(rows).to_csv(bars_path.with_suffix('.csv.tmp'),index=False)
    bars_path.with_suffix('.csv.tmp').replace(bars_path)
    ledger_path.with_suffix('.csv.tmp').replace(ledger_path)
    print(f'coverage {len(rows)//2}/{len(metas)}; not a verified full-market panel',flush=True)


if __name__=='__main__':main()
