# 2026-09-30 research snapshot review

No qualified research watch candidates; no validated buy signal. This is not a verified full-market regular-session OHLCV run.

## Coverage

- Reference-active universe on Sep28: 2,650 securities. New listings after that date are outside this universe.
- Current OHLC comparison and previous-close comparison accepted: 2,453. Rejected: 197 (135 issuer/security identity mismatches, 59 stale/intraday snapshots, one OHLC mismatch, one gap/previous-close mismatch, one untraded bar).
- Accepted with at least 520 reference bars: 2,228.
- Conservative 20-day turnover gate failed: 1,855. Classified: 373 (including both earlier watch names for explicit review).
- Anchor-containing structures: 149; alive strict anchors: 51; alive strict anchor plus strict prior-high support: 34. All 34 were NEAR, so none passed existing classic A/B structure gates. Additional research gates produced zero qualified candidates.

## Earlier watch names

| Security | Sep30 close | Existing anchor | Anchor open | Strict anchor | Strict support |
|---|---:|---|---:|---|---|
| 041190 Woori Technology Investment | 5,740 | 2026-08-25 | 6,060 | Failed | Failed |
| 234340 HectoFinancial | 22,200 | 2026-08-28 | 21,850 | Failed | Failed |

A price recovery after the first anchor breach does not revive that failed anchor. Both remain WAIT_NEW_ANCHOR, not buy recommendations.

## Sources and limits

The collector checks KIND security identity, requested date, displayed close-time timestamp and OHLC envelope. It compares Sep30 OHLC with Yahoo chart prices and Sep29 Yahoo close with KIND previous close. Sep29 OHL are not independently confirmed. Sep29 vendor volume is not certified regular-session-only; amount is explicitly estimated as low times vendor volume.

KIND's displayed timestamp alone does not certify regular-session volume: Woori's volume changed from 670,680 to 673,603 and then 673,923 while the displayed timestamp remained 15:30:20. Current KIND amount and volume therefore also have unverified session scope. Outputs deliberately do not use the CLOSED verified-overlay contract and never enter the production verified overlay. Cache write times are recorded separately from displayed closing times.

The public FinanceData daily listing cache also failed integrity/freshness checks. Sep29 Woori close 5,860 exceeded reported high 5,780; Hecto close 22,600 exceeded high 22,500. Sep30 cache values lagged the closing snapshot. No market-cap-derived repair or ambiguous NAVER fallback was used. Official KRX batch query returned LOGOUT, so regular-session historical volume collection remains unresolved.

Historical marcap data through Sep28 is reference history, not individually verified raw exchange data. Research corporate-action bridging is not official adjustment. The collector supports only this audited Sep28-to-Sep30 gap.

## Additional research gates

Existing classic classification is unchanged. Required: A/B; strict anchor and strict prior-high support; anchor earlier than Sep29; classification unchanged when recent two-day volume and amount are zero; conservative 20-day mean amount at least KRW 5bn; a later-bar retest within 3% of the prior-high level; support risk 0–10% of current price; overhead room at least 5% and reward/risk at least 2. Nearby confirmed overhead pivots are included. These are unvalidated research filters, not proprietary Dante formulas.

## Validation and artifacts

21 local unit tests passed, including security identity, stale/intraday rejection, OHLC envelope and recent-bar/collection-ledger consistency. Real inputs matched all 2,453 accepted codes and exactly two dates each. HTML includes two reviewed charts. The report, detailed signal CSV, collection ledger and recent research bars are separate dated artifacts. No trade was placed; no profitability backtest or complete regular-session market certification was performed.
