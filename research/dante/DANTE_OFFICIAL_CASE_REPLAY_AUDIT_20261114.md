# Dante official dated case replay — 2026-11-14

This is a **research-only public article/proxy comparison**, not a recovered paid indicator formula.

**Successful reproducible run:** https://github.com/namguk-lee/dante-backtest/actions/runs/43482885796
**Code:** research/dante/dante_official_example_replay.py
**Artifact:** dante-public-official-example-audit-2.zip

## Why article dates are not exact signal labels

Many publisher posts review a stock at a particular *analysis date* but mention
signals or accumulation bars which occurred *earlier*. The true original paid
screener event timestamp and negative stock-days are not published. Therefore:
- proximity to analysis dates is **NOT** sensitivity, accuracy, F1 or same-day match;
- comparisons only use historical price/volume available up to each analysis day;
- hypothetical future Odori and "previously" captured historical Odori are excluded
  from current positive denominators.

## Results with source freshness corrected

- Publisher-dated contexts: **15**.
- Contextually comparable cases: **12**.
- Cases with sufficient and timely historical OHLCV: **11**.
- Our independent proxy family fired at least once in last 20 trading days: **8/11**.
- Same-day proxy family fired: **1/11** (엘앤케이바이오 2026-05-19).
- Cases eligible under our unrelated 20-day average trading value >= KRW 5bn: **2/11**.
- Original paid Dante signal precision / recall: **UNMEASURED** (no exact original labeled outputs).

Individual nearest model events:
- 엘앤케이바이오 2026-05-19: publisher says Odori search member; our EMA5/SMA5/bottom Odori *proxies* all turn on 2026-05-19. First paid alert date unknown.
- 피엔티 2025-04-23: official says Odori trend already broken; model on 2025-04-04/10 using explicitly labeled NAVER fallback, not on analysis day.
- 비씨엔씨 2025-08-08: official reverse accumulation/EMA112; model reverse Odori and EMA112 first cross on 2025-07-15.
- 삼기에너지솔루션즈 2025-07-21: official reverse array/60MA/accumulation; model EMA112 2025-06-25 and reverse Odori 2025-07-03. Partial family overlap only.
- STX그린로지스 2026-03-09: reverse Odori model 2026-02-10/19 and EMA112 recovery 2026-02-20.
- 아이티엠반도체 2026-03-30: model 112 first recovery 2026-03-11; research reverse accumulation/concrete 2026-03-13, NAVER fallback.
- 한국선재 2026-04-20: model EMA112 first recovery 2026-04-13; actual cited anchor candle date not identical.
- 알체라, 성신양회, 공구우먼: no *fresh model trigger* in previous 20 days; official setup may already exist and needs a **state detector**.
- 레이저옵텍: observed stock episode has only 352 bars, below the research minimum 560; excluded.

## Source integrity bug and fix

The prior source's last valid price was **2022-11-18** for 피엔티
(137400) and 아이티엠반도체 (084850), despite official 2025/2026 analyses.
Earlier naive replay incorrectly compared old history as current. The
replay now **rejects histories stale by >7 days**, then requests isolated
2017–analysis-day NAVER source history via FinanceDataReader for those two.
Both historical fallback downloads worked in the successful run. Their
vendor-specific price adjustment may differ from marcap. No full production
source migration has been made.

## Next stage and deployment policy

Create independent setup-*state* indicators as well as new-*event* indicators;
validate on exact publisher signal dates and negative controls if obtainable
with permission. Compare F1, false positives per 1,000 stock-days, market
coverage, price adjustment and candidate-date staleness before evaluating
whether the research is fit for a production pull request. **No live trading,
no changes to main's production scanner, no claim to copy a paid formula.**

The artifact contains the 15 original publisher source URLs and case dates.
