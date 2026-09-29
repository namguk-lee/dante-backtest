# Public EMA224 four-month search rule — historical validation

Status: research-only discovery rule. **Not a validated entry rule.**

Public-search approximation tested:
- reference line: EMA224
- at least 80 consecutive trading sessions below EMA224 (transparent approximation of ~4 months)
- current price within EMA224 ±10%
- 20-session average turnover >= KRW 5bn

Backtest run:
- https://github.com/namguk-lee/dante-backtest/actions/runs/36632902379
- commit: bfb2030ae842dfa4a92bdde0761e86374bc78200
- 50bp cost deducted from forward returns
- splits: TRAIN 2021–2023 / VALID 2024–2025 / TEST 2026

## Base rule result

| Split | Events | 5d mean / median | 20d mean / median | 60d mean / median |
|---|---:|---:|---:|---:|
| TRAIN | 1,771 | -1.24% / -2.24% | -3.18% / -4.51% | -5.94% / -8.70% |
| VALID | 1,083 | -1.36% / -2.54% | -2.66% / -4.75% | -4.89% / -8.95% |
| TEST 2026 | 186 | -1.67% / -3.29% | -8.23% / -10.43% | -12.50% / -21.60% |
| ALL | 3,040 | -1.31% / -2.37% | -3.27% / -4.92% | -5.87% / -9.24% |

Conclusion: using the search match itself as an entry is strongly rejected.

## Add EMA112 turn + accumulation-cool context

`TURNED_UP_ACCUM_COOL` means:
- EMA112 20-session slope > 0
- at least one >=2x volume burst in the prior 20 sessions
- current volume <=1.2x the 20-session average

| Split | Events | 5d mean / median | 20d mean / median | 60d mean / median |
|---|---:|---:|---:|---:|
| TRAIN | 369 | -1.67% / -2.71% | -3.35% / -4.94% | -6.03% / -10.88% |
| VALID | 301 | -0.53% / -2.12% | -1.61% / -4.25% | -2.76% / -8.78% |
| TEST 2026 | 62 | -1.21% / -2.89% | -10.37% / -18.20% | -18.95% / -26.92% |
| ALL | 732 | -1.16% / -2.51% | -3.22% / -5.06% | -5.67% / -10.67% |

Conclusion: EMA112 turning up plus prior-volume/current-cooling context does not create a robust entry edge.

## Blue-dot BB35 context sensitivity

Adding the research BB35 blue-dot recent-5 context looks positive in the tiny 2026 slice but worsens TRAIN/VALID. This is classic regime/small-sample instability and must not be promoted.

## Live-board interpretation

The public EMA224 search rule is kept because it is useful for **discovery**, not because it predicts immediate forward returns.

Current live state labels are descriptive only:
- `TURNING_COOLED`: EMA112 has turned up, prior volume burst exists, current volume has cooled.
- `TURNING`: EMA112 has turned up but accumulation/cooling context is incomplete.
- `RAW_COOLED_CONTEXT`: long-MA search context plus cooled-after-burst, but EMA112 has not turned.
- `RAW_CONTEXT`: raw public search context only.
- `EVENT_SPIKE`: current volume itself is >=2x average; do not confuse with a quiet pullback state.

Every match keeps:
- `action_status = WATCHLIST_RESEARCH_ONLY`
- `entry_validation = NOT_VALIDATED`
- `public224_entry_evidence = SEARCH_DAY_ENTRY_BACKTEST_NEGATIVE`

## Practical consequence

Do not rank this board by historical return or label it BUY.
Use it to identify the small current set of names sitting in the publicly described long-below-224 / near-224 context, then inspect which other documented structures are also present.


## Delayed recovery -> pullback validation

A second test asked whether the direct public EMA224 search context becomes useful if entry is delayed until:

1. price first recovers EMA224 after at least 80 consecutive sessions below it,
2. then, within 20 sessions, price makes the first quiet pullback near EMA224,
3. recent closes show settlement above EMA224,
4. current volume is <= 1.2x the 20-session average.

This is a transparent research approximation, not a proprietary Dante formula.

GitHub Actions:
- run: https://github.com/namguk-lee/dante-backtest/actions/runs/36634214065
- commit tested: 9bb4b02026b202e11c04d2a4d07a7cad742207fa
- primary tolerance: 5%
- cost: 50bp deducted from forward returns

| Split | Events | 5d mean / median | 20d mean / median | 60d mean / median |
|---|---:|---:|---:|---:|
| TRAIN 2021-2023 | 347 | -1.39% / -1.73% | -4.06% / -4.77% | -8.80% / -10.32% |
| VALID 2024-2025 | 218 | -1.34% / -2.18% | -3.23% / -5.70% | -6.10% / -8.04% |
| TEST 2026 | 20 | +1.74% / -0.15% | -9.98% / -9.58% | -29.73% / -33.05% |
| ALL | 585 | -1.27% / -1.89% | -3.94% / -5.13% | -8.37% / -10.29% |

Adding the prior-volume/current-cooling context did not improve robustness. Sensitivity checks at 3%, 5%, and 8% pullback tolerances remained negative out of sample.

Conclusion:
- **Reject** generic "long below EMA224 -> reclaim -> first quiet EMA224 pullback" as a standalone entry rule.
- The positive 2026 five-day mean is based on a tiny sample and has a slightly negative median; it is followed by strongly negative 20/60-day results.
- Keep the public 224 rule as a discovery context only.
- Continue testing richer public structure (Bowl duration, anchor/accumulation, recovery, and post-pullback re-acceleration) rather than optimizing the 224 tolerance.
