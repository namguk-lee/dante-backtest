# 단테 공식 분석 사례 3차 검증 — 상태 vs 신호 / 일반 시장대조 (2026-11-14)

Status: **research-only**. Official paid proprietary indicator match is unmeasured. Existing main scanners and trading rules unchanged.

## Reproducibility

- Independent code: research/dante/dante_official_state_vs_event.py
- Workflow: .github/workflows/dante-official-state-controls.yml
- Latest successful run: https://github.com/namguk-lee/dante-backtest/actions/runs/43522863819
- 2016–2026 KOSPI/KOSDAQ issuer-episode historic panel, daily OHLCV known as of the publicly cited analysis date.
- 15 official contexts documented in prior audit. 3 excluded: retrospective-only/hypothetical and not current signal labels. 12 contextually comparable, 11 with sufficient EMA448 and timely history; 1 (레이저옵텍, 199550) had 352 bars, therefore not comparable for the full long-EMA proxy.

## Main numerical findings

| Official-analysis family | Cases with broad as-of state | Had a new comparable proxy event in previous 20 trading sessions | Same-date, same-market + broad-turnover matched **unlabeled** ordinary stock-days also showing state |
|---|---:|---:|---:|
| ODORI | 2/2 | 2/2 | 16/40 = 40.0% |
| EMA112 recovery/hold | 1/1 | 0/1 | 8/20 = 40.0% |
| Reverse-array + accumulation + base | 5/5 | 5/5 | 26/100 = 26.0% |
| Bowl/base | 2/2 | 0/2 | 25/40 = 62.5% |
| Anchor holding | 1/1 | 1/1 | 9/20 = 45.0% |
| **Totals** | **11/11** | **8/11** | **84/220 = 38.2%** |

**Critical: 11/11 is NOT the paid original indicator recall or 100% accuracy.**
The official source contexts were used when defining our transparent diagnostic categories; the sample is NOT independent held-out validation. Controls are ordinary date/exchange/turnover matched **unlabeled** stocks, not certified signal negatives. Selected controls per case: 20; turnover match broad factor 0.1–10x actual prior20 average; 220 stock-days drawn deterministically with SHA-256.
No proprietary signal precision/recall/F1, predictive trading win rate, p-value, or actual false-positive rate is warranted from these samples.

### Missed **states** rescued by separating state from new-event detection

- 알체라: EMA112 recovery/holding state without a fresh event during the previous 20 trading days.
- 성신양회: bowl/base state without a new trigger in the same lookback.
- 공구우먼: bowl/base state without a fresh trigger in the same lookback.

All three are **generic structural interpretations**; this does not prove matching paid premium color-coded indicators.

## Liquidity filter sensitivity (our cutoff, NOT Dante official settings)

| Prior20 average traded amount | Official contexts retained | Recent independent event contexts retained |
|---|---:|---:|
| No cutoff | 11 / 11 | 8 |
| ≥ KRW 500m (5억) | 9 / 11 | 7 |
| ≥ KRW 1bn (10억) | 8 / 11 | 6 |
| ≥ KRW 2bn (20억) | 4 / 11 | 3 |
| ≥ KRW 5bn (50억) | 2 / 11 | 2 |

Never simply remove the 5bn production liquidity screen; classify **structure** separately from **execution/liquidity risk** and measure spreads, slippage and tradability before adjusting the live pipeline.

## Predeclared research definition (NOT proprietary)

- EMA5/SMA5 upward turn within previous20 and price above EMA5 for Odori state.
- Price above EMA112 after an EMA112 crossing in prior80 for 112 held state.
- Reverse-array 112<224<448 within prior120, accumulation candle within prior120 (≥1.8x preceding20 volume; positive candle body≥2.5%; close position≥65%), stable base within prior30, close above EMA112 or EMA60 for accumulation/concrete state.
- Bowl state: at least one stable 40–90 day base, preceding peak-to-trough decline ≥20%, current distance to EMA224 between -15% and +20%.
- Stable base: closing 10th-to-90th percentile price width ≤30%, window net change -18% to +20%, over the last 40/60/90 bars ending in prior30.
- Anchor still holds when price not under most recent observed anchor open.
- Same-day comparisons use no OHLCV after analysis date.
- Names/codes are retained even when a historical episode lacks 448 sessions.
- Stock history corrupted/stale in primary source for 피엔티 and 아이티엠반도체; isolated FinanceDataReader NAVER historical recovery, with adjustment-source differences noted. No production data pipeline change.

## Independent validation gate before production

1. Obtain truly dated **original** screener outputs and a sizable set of actual non-alert days for candidate indicators, legally accessible from the owner.
2. Do not tune thresholds with the current 11 officially labeled contexts and then call the same set an out-of-sample test. Reserve new dates and issuers as untouched controls.
3. Compare **event precision/recall/F1** using ground truth; measure state prevalence separately, with per-market false alerts per 1000 stock-days and total detection volume per day.
4. After technical fidelity is verified, assess post-signal 5/10/20d win rates, medians, MFE/MAE, realistic fees, slippage and failed issuer/IPO data coverage.
5. Until then all results are research, independent of recommendation production A/B scans and trading (no orders).

Official article sources:
- https://contents.premium.naver.com/jusikdante/jusikdante1/contents/260519170521976rz (엘앤케이바이오 2026-05-19)
- https://contents.premium.naver.com/jusikdante/jusikdante1/contents/251002164439202kg (비씨엔씨 2025-08-08)
- https://contents.premium.naver.com/jusikdante/jusikdante1/contents/240903191444551ao (EMA chart configuration)

CSV artifacts: case_vs_market_prevalence.csv, liquidity_gate_sensitivity.csv, official_asof_state_features.csv, matched_unlabeled_market_controls.csv.
