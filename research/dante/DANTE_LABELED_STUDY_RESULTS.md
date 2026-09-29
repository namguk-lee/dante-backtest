# Dante labeled public-example study — first empirical pass

Status: research-only. Live scanners were **not** changed by this study.

Run:
- GitHub Actions: https://github.com/namguk-lee/dante-backtest/actions/runs/36553973658
- 22 labeled public/official examples
- 22 matched to historical KOSPI/KOSDAQ data
- 0 misses
- Every feature is computed using market data available **on or before the published analysis date**.

## 1. Main correction: EMA224 slope is not a universal gate

Among the 18 examples labeled as not hitting the published stop:
- 88.9% were above EMA112 on the analysis date.
- 77.8% were above EMA224.
- 72.2% had a positive EMA112 20-session slope.
- Only 44.4% had a positive EMA224 20-session slope.

Medians:
- success dist to EMA112: +7.60%
- stop/mixed dist to EMA112: +4.02%
- success dist to EMA224: +3.89%
- stop/mixed dist to EMA224: -3.88%
- success absolute 112/224 gap: 3.76%
- stop/mixed gap: 6.71%
- success EMA112 slope20: +1.01%
- stop/mixed EMA112 slope20: -0.78%
- success EMA224 slope20: **-0.81%**
- stop/mixed EMA224 slope20: -1.83%

Interpretation:
- Public success examples often have price/EMA112 turning first while EMA224 is still mildly falling.
- This matches the publicly described 256 and MA-hit logic.
- Therefore the Classic V2 requirement that EMA224 must already be rising is appropriate only for a stricter late/confirmed structure, not for all Dante techniques.

## 2. Volume pattern: prior accumulation, current quiet pullback

Among the 18 non-stop examples:
- 66.7% had analysis-day volume below the 20-session average.
- 88.9% had at least one >2x 20-session-average volume day within the previous 20 sessions.
- median analysis-day volume ratio = 0.89x
- median maximum volume ratio in the prior 20 sessions = 6.75x

Interpretation:
- A recurring public-example structure is: **strong volume/accumulation first -> quieter correction/settlement later**.
- Requiring the entry/analysis day itself to have explosive volume would miss many labeled examples.

## 3. Reverse-array context is common, but not sufficient

- 83.3% of non-stop examples had EMA112 < EMA224 < EMA448 at the analysis date.
- 75.0% of stop/mixed examples also had the same reverse-array condition.

Interpretation:
- Reverse array is a common context for the studied setups.
- It is not a sufficient ranking signal by itself.

## 4. Blue-dot candidate formulas: no clean match yet

Candidate distances were tested against:
- Bollinger upper 20 / 2
- Bollinger upper 35 / 2
- Bollinger upper 40 / 2
- Bollinger upper 20 / 2 shifted by 26 sessions

The labeled set contains 19 blue-dot examples and only 3 examples without a named blue-dot signal, so it is highly imbalanced.

None of the simple candidates cleanly separated blue-dot vs non-blue-dot labels. In particular:
- the 35-period recreation did not consistently outperform 20 or 40
- the shifted 20/2 candidate also did not show a convincing match

Conclusion:
- Do **not** promote any internet Bollinger recreation to an “official blue-dot formula.”
- Next study needs more official negative/control examples and, ideally, chart-image geometry.

## 5. Share / 지분 1:1: simple price-distance mirroring is rejected

We tested a simple candidate based on the maximum downside distance below MA60/112/224 before the latest upward cross versus the current upside distance above the same MA.

This does not reproduce official labeled examples:
- Messe Esang, officially described as 224-based 1:1: MA224 candidate ≈ 0.70
- BCNC, officially described as 112-based 1:1: MA112 candidate ≈ 0.39
- Parton, officially described as 112-based 1:1: MA112 candidate ≈ 0.05

Conclusion:
- The published 1:1 is very unlikely to be our simple current-price-vs-prior-low mirror formula.
- Likely candidates include swing-box geometry, wave amplitude, area/territory comparison, or another construction.
- Do not implement the current mirror candidate as the Dante formula.

Official descriptions:
- CP System: https://contents2.premium.naver.com/jusikdante/jusikdante1/contents/250919151431347pa
- Messe Esang: https://contents.premium.naver.com/jusikdante/jusikdante1/contents/250905133030841pp
- BCNC: https://contents.premium.naver.com/jusikdante/jusikdante1/contents/251002164439202kg

## 6. Proprietary signals do not guarantee success

Watermelon and Blue-dot also appear in labeled stop cases such as Ilyeon Pharmaceutical and STX Green Logis. Rainbow appears in mixed/stop-out examples as well.

## 7. Architectural consequence

Do not keep expanding one serial Classic A/B gate. Future research architecture should be parallel:
- public_256_long
- public_ma_hit_112_to_224
- public_ma_hit_224_to_448
- public_bowl3
- public_reverse112
- public_anchor
- public_concrete
- research_share_1to1_*
- research_blue_dot_proxy_family
- later research_watermelon_proxy
- later research_rainbow_proxy

Output should become a signal matrix/confluence report rather than one serial grade.

## 8. Important limitations

- Official examples are positively selected cases.
- Blue-dot 19/22 and Watermelon 20/22 are heavily imbalanced.
- Indicator not named in an article does not prove it was absent.
- Only four examples are stop/mixed.
- The study is useful for rejecting poor proxy ideas, but cannot establish a proprietary formula yet.

## Next research before any live-scanner change

1. Expand labeled examples, especially explicit failures and clearly absent-signal controls.
2. Reconstruct 지분 1:1 from chart geometry, not simple price distance.
3. Collect blue-dot dated screenshots and compare band/shift families.
4. Use matched same-date controls to test discrimination beyond generic rebound structures.
5. Only then add research-only proxy detectors.