# Dante Stage 4 — does a long base + EMA224 breakout survive? (2026-11-14)

**Research only.** Outcome is the public-Dante-inspired V4 simulator (2R target/structural stop at next adjusted open), **NOT** official paid indicator performance nor an actual executed strategy. All threshold experiments here are retrospective and do not justify trading.

## Reproducible study

- V4 full-market signal generator: research/dante/dante_base_v4_daily_backtest.py (imported unchanged from earlier audited research branch agent/dante-base-v3-validation-20261113).
- Stage 4 comparison and fixed train-only classification: research/dante/dante_v4_stage4_success_validation.py.
- CI workflow: .github/workflows/dante-v4-stage4-validation.yml.
- Input to the local first replay: dante_v4_all_signals_2018_2026.csv (previously generated full-market V4 research artifact).
- Segment: profile B60_T50EOK, setup EMA224_BREAKOUT. Price history includes KOSPI/KOSDAQ, with issuer-episode separated historical data and some provider-research price adjustments.

### Data hygiene correction — 6 cases censored

In the prior work 1,299 events = 329 target + 763 stop + 201 actual 20-session TIMEOUT + **6 immature outcomes**. The 6 must NOT be counted as TIMEOUT, giving 1,293 fully observed outcomes. Mature target 329/1293=25.44%; mature stop 763/1293=59.01%; timeout 201/1293=15.55%.

### Study filters (not official Dante parameters)

"Stable + volume + EMA224-near": completed base span (q90/q10-1) <=25%, completed base first/last change -5..+15%, event volume/mean20 >=2, close/EMA224 -1..+5%. This experiment was chosen after seeing earlier data.

| Split | Baseline mature / target / avg net 20d | Combined mature / target / avg net 20d |
|---|---|---|
| TRAIN 2018–21 | 524 / 25.00% / -0.63% | 187 / 25.67% / -0.54% |
| VALID 2022–24 | 592 / 25.00% / -1.14% | 216 / 27.78% / -0.54% |
| TEST 2025–26 | 177 / 28.25% / -0.83% | 53 / 32.08% / -0.29% |

Mean net outcomes remain **negative** in all periods. Bootstrap by issuer code, 1,000 fixed-seed resamples, target-rate improvement (combined minus original, **percentage points**):
- TRAIN +0.67pp, 95% interval -4.44..+5.81pp.
- VALID +2.78pp, 95% interval -2.14..+7.82pp.
- TEST +3.83pp, 95% interval -6.51..+14.14pp.
All confidence intervals include zero; no robust selectivity is established.

### Prespecified fixed-feature logistic discrimination

Logistic regression with feature-only values known at signal CLOSE:
completed base width, base net change, previous peak decline, signal EMA224 gap, relative volume, number of last 80 days below EMA224. Normalize/median imputations fit to 2018–21 only; fixed C=0.5, no hyperparameter search. **Next-day entry open/stop-risk distance are NOT included as predictors.** Labels: 2R target first = 1; stop or mature time-out = 0.

| Date partition | Mature N | ROC AUC | Average precision | top TRAIN-score quintile 2R hit | top mean net20d |
|---|---:|---:|---:|---:|---:|
| TRAIN 2018–21 | 524 | 0.607 | 0.324 | 35.2% | +1.29% |
| VALID 2022–24 | 592 | 0.551 | 0.312 | 29.9% | -0.99% |
| TEST 2025–26 | 177 | **0.462** | 0.336 | 33.3% | -0.67% |

This shows **no stable prospective prediction**. TEST is only a time-based partition, not a pristine independent out-of-sample: this team's historical development previously inspected the same periods. AUC <0.5 does not prove the opposite trade succeeds.

### Repeated-signal sensitivity

A crude >=90 calendar day cooldown per same code reduces baseline total events to 1,233, mature 1,228 and mature target hit ~25.2% versus original 25.4%; use this only as a sensitivity check. It is not exactly a 60-trading-day cooldown.

## Limitations and next steps

1. The original proxy is **EMA224 first breakout**, not all true Dante "reverse MA accumulation / concrete" stages.
2. Trading-value >=KRW5bn gate excludes many genuine official context examples (2 of 11 source articles passed it). Keep liquidity-risk ranking separate from structural pattern discovery.
3. Original "state matches" had high background prevalence in unlabeled normal stocks (84/220=38.2%): official example matching does not certify predictive efficacy.
4. Price-source integrity previously failed for two market symbols, restored only in the article replay branch via tagged Naver. Reconcile all stale/adjusted price episodes before promotion.
5. Validate genuine future KRX trading dates with frozen as-of signal features, market/sector regime control, and first completed 1/3/5 day post-signal behavior. Never condition an as-of-close alert on future candles.
6. Original paid indicators Blue Dot/Watermelon/Rainbow formula and first occurrence timestamps remain unknown; do not state proprietary match accuracy.
7. This study never places orders. Production main scanning/18:30 PDF automation remains unchanged. Workflow currently not separately confirmed completed; do not claim the new CI execution passed without an actual GitHub run.

**Research decision:** DO NOT MERGE model coefficients or these cutoff choices as production buy/recommendation conditions.
