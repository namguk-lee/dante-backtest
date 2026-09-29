# Dante share 1:1 research — empirical result

Status: **research proxy only**. This is not claimed to be Dante's proprietary/exact formula.

## What was tested

Public recaps that explicitly describe a 1:1 share setup were reconstructed at their stated analysis dates:

- Messe Esang — 2025-06-17 — EMA224
- CP System — 2025-08-01 — EMA60
- BCNC — 2025-08-08 — EMA112
- Saevitchem — 2025-08-01 — EMA112
- Partron — 2025-11-17 — EMA112

All calculations use only data available on or before the stated analysis date.

## Rejected candidates

These hypotheses did not reproduce the labeled examples consistently:

1. Latest up-cross of the reference MA.
2. Fixed-window candle-count 50:50 using 60/112/224-session windows.
3. Current price distance above MA divided by a generic historical maximum distance below MA.
4. Latest local trough depth below MA versus current height above MA.

The naive `ratio >= 1` rule was especially non-selective: in same-date market controls, roughly 66% to 89% of eligible names satisfied it depending on date/MA.

## Best current structural selector

Select the **latest structural upward recovery of the chosen reference MA that was preceded by at least 30 consecutive trading sessions below that MA**.

For that structurally selected cycle, two measurements were most useful:

### A. Recovery-height ratio

`long_ratio = upside from reference MA after recovery / prior sell-side excursion below reference MA`

Official labeled values:

| Case | MA | Pre-below days | Recovery ratio |
|---|---:|---:|---:|
| Messe Esang | 224 | 42 | 0.727 |
| CP System | 60 | 192 | 1.672 |
| BCNC | 112 | 39 | 0.923 |
| Saevitchem | 112 | 467 | 1.011 |
| Partron | 112 | 137 | 1.135 |

Four of five fall inside 0.65–1.35. All five fall inside the broader 0.65–1.75 research band.

### B. Cycle mean-distance balance

Define a structural sell->buy cycle from the prior meaningful down-cross through the analysis date, then calculate:

`cycle_mean_dist_ratio = mean normalized distance above MA / mean normalized distance below MA`

Official labeled values:

| Case | Mean-distance ratio |
|---|---:|
| Messe Esang | 0.688 |
| CP System | 1.303 |
| BCNC | 0.823 |
| Saevitchem | 1.250 |
| Partron | 0.876 |

All five are inside 0.65–1.35.

## Matched-market control result

Run: https://github.com/namguk-lee/dante-backtest/actions/runs/36562377130

Same-date KOSPI/KOSDAQ controls show the proxy is not merely a universal rebound pattern:

- Recovery-height ratio within 0.65–1.35 occurred in about **18.8%–26.6%** of eligible controls.
- Cycle mean-distance ratio within 0.65–1.35 occurred in about **29.4%–34.6%**.
- The relaxed combined rule used only for hypothesis screening:
  - `long_ratio` 0.65–1.75
  - `cycle_mean_dist_ratio` 0.65–1.35
  appeared in about **13.4%–19.5%** of comparable controls.

Composite closeness-to-1 percentile for the five official cases (higher = closer to balanced than more market controls):

- Messe Esang: 84.6 percentile
- CP System: 77.2 percentile
- BCNC: 95.0 percentile
- Saevitchem: 95.4 percentile
- Partron: 98.0 percentile

## Interpretation

The strongest current public-data interpretation is not a fixed candle-count rule and not a single local-low mirror.

A more plausible approximation is:

1. choose the contextually relevant long MA (60/112/224),
2. identify a meaningful sell-side phase that persisted below it,
3. identify the structural recovery above it,
4. compare the prior sell-side excursion/territory with the new buy-side excursion/territory,
5. treat near-balance as a transition condition rather than an entry signal by itself.

This is directionally consistent with public descriptions of the MA as a boundary between buy-side and sell-side territory and of the balance becoming roughly 1:1 before buy-side control strengthens.

## What this does NOT establish

- It does not prove the exact proprietary construction of the white 1:1 box.
- The five explicit positive cases are a small sample.
- Thresholds 0.65/1.35/1.75 are research bands selected for robustness testing, not published Dante constants.
- The proxy must not be named `Dante exact 1:1` in production.

## Next validation

1. Render the structurally selected long-below cycle on the five dated charts.
2. Compare its chosen boundaries visually with the white boxes in the public recap images.
3. Add additional explicit 1:1 positive and pre-1:1 negative examples.
4. Only if visual and expanded-sample validation holds, expose it as `research_share_1to1_proxy` in the parallel signal matrix.
5. Do not use it as a standalone buy signal.