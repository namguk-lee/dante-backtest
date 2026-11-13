#!/usr/bin/env python3
"""Research-only post-hoc sensitivity audit of existing V4 events.

This does NOT infer proprietary Dante formulas or create live trade signals.
Features are known as of each signal date, but scenarios were selected after
seeing historical outcomes. Existing V4 simulated exits are reused. Results
are thus retrospective diagnostics, not a clean prospective test.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def summarize(source: pd.DataFrame) -> pd.DataFrame:
    d=source[source.profile.isin(['B60_T50EOK','B90_T50EOK'])].copy()
    first=lambda q:q.setup.eq('EMA224_BREAKOUT')
    b60=lambda q:q.profile.eq('B60_T50EOK') & first(q)
    b90=lambda q:q.profile.eq('B90_T50EOK') & first(q)
    near=lambda q:q.dist224_pct.between(-1,5)
    stable=lambda q:q.base_range_pct.le(25) & q.base_change_pct.between(-5,15)
    heavy=lambda q:q.vol_ratio.ge(2.0)
    rules={
      'B60_first224':b60,
      'B60_224near':lambda q:b60(q)&near(q),
      'B60_stable':lambda q:b60(q)&stable(q),
      'B60_volume2':lambda q:b60(q)&heavy(q),
      'B60_stable_volume2_near':lambda q:b60(q)&stable(q)&heavy(q)&near(q),
      'B90_first224':b90,
      'B90_stable_volume2_near':lambda q:b90(q)&stable(q)&heavy(q)&near(q),
    }
    output=[]
    for name,check in rules.items():
        mask=check(d)
        if not mask.index.equals(d.index):
            raise ValueError('Signal mask index mismatch')
        for phase in ('TRAIN','VALID','TEST'):
            x=d[mask & d.split.eq(phase)]
            m=x.dropna(subset=['net_20d'])
            output.append(dict(
                rule=name,phase=phase,events=len(x),mature20=len(m),
                avg_net20_pct=round(m.net_20d.mean(),3),
                median_net20_pct=round(m.net_20d.median(),3),
                win20_pct=round(100*(m.net_20d>0).mean(),1),
                stop20_pct=round(100*m.outcome_20d.fillna('').str.startswith('STOP').mean(),1),
                year2025=int((x.year==2025).sum()),year2026=int((x.year==2026).sum())
            ))
    return pd.DataFrame(output)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--signals',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    d=pd.read_csv(args.signals,encoding='utf-8-sig',dtype={'code':str},low_memory=False)
    required={'profile','setup','dist224_pct','base_range_pct','base_change_pct',
              'vol_ratio','split','net_20d','outcome_20d','year'}
    missing=required-set(d.columns)
    if missing:raise RuntimeError(f'Missing V4 columns: {sorted(missing)}')
    out=summarize(d)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(args.out,index=False,encoding='utf-8-sig')
    print(out.to_string(index=False))
    assert (out.groupby('rule').phase.nunique()==3).all()

if __name__=='__main__':main()
