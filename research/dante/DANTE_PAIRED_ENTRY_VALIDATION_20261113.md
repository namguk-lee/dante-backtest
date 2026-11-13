# Dante EMA224 first-breakout vs pullback/rebreakout paired comparison

Date: 2026-11-13 (data through 2026-11-12).
Status: **research only, not validated for buy recommendations**.
Workflow: https://github.com/namguk-lee/dante-backtest/actions/runs/43414137450

## Design

- Same source V4 EMA224 breakout events per selected 40/60/90-day stabilized-base profile with 20-day average trading value >= KRW 5bn; **deduplicate stock signals per profile within 60 sessions**.
- First-breakout arm = V4 original simulated next-session adjusted open entry, structural stop, 2R take profit, round-trip 40bp costs.
- Waiting arm = post-breakout 2–12 session near-EMA224 pullback with falling volume and <=22% decline, followed within 8 sessions by close above prior 3- or 5-session high and the EMA112/224, with volume confirmation; then entry **next open**. Gap-entry and risk bounds checked.
- Both arms have identical **parent signal start date** and end-of-study deadline (20 or 60 sessions). Waiting strategy earns **0% cash return when no signal**; evaluate all eligible parent events (intention-to-treat) as primary measure. Filled-subset results are shown separately to avoid falsely improving performance by excluding unfilled opportunities.
- Train 2018-2021, validation 2022-2024, already-reviewed test 2025-2026 (NOT an untouched OOS sample).
- No execution, no broker integration. Adjusted historic OHLC and fee model are proxies, not live realizable returns.

## Main 20-session results (units: percent; averages after 40bp cost)

| Base | Waiting | Split | Parent count | Confirmed | First-breakout avg | Wait/cash ITT avg | First-breakout avg on *filled-only* subset | Delayed-entry avg on *filled-only* subset |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 60d | 3d high | TRAIN | 498 | 113 | -0.762 | -0.072 | +3.432 | -0.325 |
| 60d | 3d high | VALID | 560 | 109 | -1.105 | -0.308 | +2.264 | -1.605 |
| 60d | 3d high | TEST | 164 | 44 | -1.019 | -0.234 | +3.020 | -0.856 |
| 90d | 3d high | TRAIN | 330 | 92 | -0.651 | -0.466 | +2.158 | -1.703 |
| 90d | 3d high | VALID | 415 | 90 | -0.754 | -0.627 | +1.406 | -2.944 |
| 90d | 3d high | TEST | 105 | 34 | -0.514 | -0.309 | +4.129 | -0.936 |

NOTE: Confirmed counts include a few immature end-date examples; realized filled-only average uses only mature cases. Report CSV has exact denominators.
- A cash-waiting strategy avoids many losing trades, leading to smaller aggregate loss; **this is NOT evidence that pullback/rebreakout entries themselves are superior**.
- The actual filled delayed trades underperform the initial-breakout arm on the **same confirmed subset** in all six 60/90-day 3D split cohorts above.
- 90d/3D VALID **60-session** waiting strategy is worse than first-breakout average (-0.983 vs -0.881); no consistent advantage across horizons.

## Actual opportunities per year

| Base / 3D rebreak | 2025 original breakouts | 2025 confirmation count | 2026 breakouts through Nov 12 | 2026 confirmations through Nov 12 |
|---|---:|---:|---:|---:|
| 40d | 143 | 36 | 78 | 17 |
| 60d | 112 | 33 | 52 | 11 |
| 90d | 71 | 25 | 34 | 9 |

Cross-base profiles overlap: do NOT add counts together.

## Decision

Keep research code in PR only. Do **not** merge into main or re-label RESEARCH_ONLY as a buy signal. Next experiments must be pre-registered (e.g. partial scale-out, ATR-backed stop, EMA112 slope, longer support confirmation) and evaluated on a separate, untouched future period; avoid repeated optimizing on the same historical outcomes.

Files produced:
- paired_comparison/paired_trades.csv
- paired_comparison/summary_by_split.csv
- paired_comparison/summary_by_year.csv
- paired_comparison/summary_by_variant.csv
- paired_comparison/summary_by_market.csv

Study code: research/dante/dante_breakout_vs_rebreak_paired.py
