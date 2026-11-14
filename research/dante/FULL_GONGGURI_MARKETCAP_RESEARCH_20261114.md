# Marketcap-relative complete Gongguri B60/B90 study — 2026-11-14

**Research only, not a paid Dante indicator or buy order.**

Verified historical run: https://github.com/namguk-lee/dante-backtest/actions/runs/43543436161

## Structure applied

A complete Bowl2 (Gongguri) requires at least 60 sessions, at least as many sessions as the prior decline (Bowl1), prior peak decline at least 20%, >=80 previous EMA224-under sessions, 10th-to-90th percentile core-base width <=35%, and core close net movement within +/-20%. Bowl 90 sessions are listed separately. AFTER this base, classify volume anchor, survival of anchor-open support, EMA224 and concrete resistance breakout/hold. These are independent numeric proxies, not proven proprietary rules.

Legacy anchor gate = current-inclusive 20 trading days KRW 5bn average amount. Capital-aware gate = previous 20 sessions' average amount and previous market cap both above size-class thresholds (small <KRW100bn: KRW100m and 0.1%, mid <KRW1tn: KRW300m and 0.04%, large >=KRW1tn: KRW1bn and 0.015%). These are experimental thresholds.

## As-of 2026-11-13 actual daily price bars

Completed Gongguri research structures total **52 distinct symbols**:
- BASE_ONLY: 30 symbols with no accepted anchor.
- BOTH old+cap-anchor: 7.
- CAP_ONLY: 15.
- LEGACY_ONLY: 0.

B60 32, B90 20. In CAP_ONLY, **only 3 retain anchor or resistance hold**: 지역난방공사 (B113; anchor held), 대동금속 (B107; concrete held), 코셈 (B65; concrete held). The other 12 are anchor-unheld and should not appear as a confirmed breakout. In BOTH, support held in 루닛 (B81), 대명에너지 (B94), 크래프톤 (B100). None is a current confirmed paid Dante signal nor necessarily a new event on 2026-11-13.

Cap provenance: all 52 final session market caps are labeled LAST_REPORTED_SHARES_ESTIMATE rather than KRX verified same-day cap, because Naver appended the final price date without cap/share data. Cap-relative volume anchors use the actual historical KRX_MARCAP_WON_CHECKED prior information, not the implied final-session market cap. Display the estimates clearly and flag ordinary-share vs preferred-share risks.

## Historical EMA224 first breakout 2R simulator (20 sessions)

Same identical B60 technical rules and original stop-first gap-aware next-open entry with 0.4 percentage point cost. Mean net 20-session performance by liquidity gate:

| Historical period | Legacy 50e mature N / mean | Market-cap normalized mature N / mean |
|---|---|---|
| 2018-2021 | 524 / -0.63% | 2782 / -0.20% |
| 2022-2024 | 592 / -1.14% | 2354 / -1.14% |
| 2025-2026 | 179 / -0.79% | 1252 / -0.24% |

All three marketcap-normalized means remain negative, even though more stocks are included. The V4 simulator checks next-day OPEN to determine allowable stop risk and thus is NOT a pure as-of-current-close signal checker; historical labels are censored until 20 trading bars mature. Correlated tickers/B60 vs B90 and re-used historical periods limit statistical generalization. Do not claim future profitability or an original proprietary formula match.

## Implementation

- research/dante/dante_cap_full_bowl_daily.py: complete Gongguri structure, A/D gate delta and full-base/anchor/concrete stages. Includes thick HLC3-volume **daily** AVWAP anchored to actual base start, not intraday VWAP or paid Blue Dot.
- research/dante/dante_cap_bowl_historical.py: 2018-2026 B60/B90 20/60-day 2R gap-aware study under both gates.
- research/dante/dante_full_bowl_from_classic_results.py: lightweight reporting of genuine completed Gongguri from existing classic A/D artifacts.
- research/dante/test_cap_full_bowl.py and test_full_bowl_postprocess.py: state, source, duplicate, censored outcome and no-false-new-alert tests.
- .github/workflows/dante-full-bowl-cap-validation.yml: full historical research and source tests.
- .github/workflows/dante-full-gongguri-daily.yml: weekday 17:10 Korea time isolated research scan; no duplicate user notification, existing 18:30 ChatGPT PDF workflow preserved.

Only integrate outcomes after full CI passes and verify actual current trade date. Missing source must be marked missing, not silently regarded as zero new signals. No automated trades.
