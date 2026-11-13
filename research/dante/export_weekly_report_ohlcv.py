#!/usr/bin/env python3
"""Export dated, auditable daily OHLCV for the 2026-11-09...13 Dante PDF.

This exports REAL daily historical OHLCV and research-adjusted OHLC. The PDF
can calculate anchored VWAP as cumulative typical-price*volume / volume.
Never infer VWAP from a candlestick PNG or call daily AVWAP intraday VWAP.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from dante_classic_scanner import load

CODES=[
    "036460","282720","002320","011200","005830",
    "267850","272450","003230","036710","204270",
    "002990","399720",
]
END=pd.Timestamp("2026-11-13")

def main():
    out=Path("dante_pdf_prices")
    out.mkdir(exist_ok=True,parents=True)
    x=pd.concat([load("weekly_market/ko_eod.parquet","KO"),
                 load("weekly_market/kq_eod.parquet","KQ")],ignore_index=True)
    result=[];summary=[]
    for code in CODES:
        p=x[x.code.eq(code)].sort_values("date").copy()
        p=p[p.date.le(END)]
        if p.empty:raise RuntimeError(f"No bars for code {code}")
        # If a code is re-used or issuer changes, keep only most recent episode.
        latest=p.date.max()
        episode=str(p[p.date.eq(latest)].series_id.iloc[-1])
        g=p[p.series_id.astype(str).eq(episode)].sort_values("date").copy()
        if g.date.max()<pd.Timestamp("2026-11-10"):
            raise RuntimeError(f"Stale data {code}: {g.date.max()}")
        if len(g)<181:
            raise RuntimeError(f"Insufficient history {code}: {len(g)}")
        factor=(g.adjusted_close/g.close).replace([np.inf,-np.inf],np.nan).ffill().bfill()
        g["adj_open"]=g.open*factor
        g["adj_high"]=g.high*factor
        g["adj_low"]=g.low*factor
        for n in (112,224,448):
            g[f"ema{n}"]=g.adjusted_close.ewm(span=n,adjust=False,min_periods=n).mean()
        g=g[g.date.ge(pd.Timestamp("2026-01-01"))].copy()
        columns=["date","code","name","exchange","open","high","low","close",
                 "adj_open","adj_high","adj_low","adjusted_close","volume",
                 "amount","ema112","ema224","ema448"]
        result.append(g[columns])
        summary.append(dict(code=code,series_id=episode,first=g.date.min(),
                            last=g.date.max(),rows=len(g),
                            last_raw_close=g.close.iloc[-1]))
        print(code,"last",g.date.max(),"rows",len(g),flush=True)
    data=pd.concat(result,ignore_index=True).sort_values(["code","date"])
    data.to_csv(out/"selected_12_ohlcv.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(summary).to_csv(out/"coverage.csv",index=False,encoding="utf-8-sig")
    (out/"DATA_METHOD.txt").write_text(
        "Market source: FinanceData/marcap; 2026 missing recent bars may be appended\n"
        "from Naver Finance via collect_marcap_full_market.py. Prices are RESEARCH\n"
        "data. adj OHLC derives historical adjusted-close ratio, which is not\n"
        "a brokerage vendor's officially adjusted chart.\n"
        "Use anchored VWAP from actual daily HLC typical price * daily volume,\n"
        "anchored at pre-event base start. This is a daily AVWAP approximation;\n"
        "cannot construct true intraday session VWAP without tick/minute data.\n",
        encoding="utf-8")
    print("DONE",len(data),"rows",flush=True)

if __name__=="__main__":
    main()
