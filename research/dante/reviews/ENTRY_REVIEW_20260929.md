# 2026-09-29 entry review (performed 2026-09-30 KST)

Both prior watchlist recommendations are downgraded to **WAIT_NEW_ANCHOR**.
These are existing classic research rules, not validated buy rules or exact
proprietary Dante indicator formulas.

| Code | Anchor | Anchor open | First later close below anchor | That close | Strict prior-high support first lost |
|---|---|---:|---|---:|---|
| 041190 | 2026-08-25 | 6,060 | 2026-09-02 | 5,840 | 2026-09-16 (5,250 level) |
| 234340 | 2026-08-28 | 21,850 | 2026-09-16 | 18,410 | 2026-08-31 (23,250 level) |

The broad matrix still reports 112 recovery for 041190 and bowl structure for
234340. Its 2% tolerance concrete approximation still flags 041190: its
5,180/5,240 closes were below 5,250 but above 98% of it. This is intentionally
shown beside the stricter existing classic rule, not counted as independent
confirmation. A later recovery does not erase the anchor's failure.

## Price-source finding

On 2026-09-30, NAVER legacy price/list, front-api daily list, and explicit
exchangeType=KRX chart responses gave 2026-09-29 closes 5,840 and 22,600.
MK/Investing regular-session references gave 5,750 and 22,200; KIND's
2026-09-30 previous-close fields independently confirmed 5,750 and 22,200.
Even the KRX-labelled response disagreed, so **the cause is unresolved**.
Do not assert that the discrepancy is necessarily caused by NXT.

The old automatic NAVER overlay is disabled. `--krx-overlay` accepts reviewed
closed KRX rows with provenance and chronology/integrity checks. This is an
input contract, not automated verification of a provider's numerical data.
`verification_scope` specifies what was actually independently checked.
The two-row ledger verifies only closes independently; OHLCV is from the MK
reference and amount is rounded to million KRW. Prior marcap history has not
been independently verified bar by bar. This is a two-stock review, not a
new full-market ranking. Default collection ends at the Seoul closed-date
cutoff and exact-freshness workflow stops when reference data is delayed.
A partial overlay cannot certify a fresh full-market panel.

## Reproduction

```bash
python research/dante/dante_entry_review.py \
  --panel review_panel.parquet --end 2026-09-29 \
  --codes 041190 234340 --out entry_review
python -m unittest discover -s research/dante -p 'test_*.py' -v
```

Use the corrected marcap loader for 2022–2026, select these two codes, append
`krx_verified_20260929.csv`, and apply the existing episode and research
corporate-action adjustment functions to create `review_panel.parquet`.
Do not overwrite historical bars with different-source snapshots. Review
truncates all input at the analysis date and excludes today's intraday bar.

No complete full-market KRX freshening provider or new expectancy backtest
was established in this step. No executable buy signal is emitted.
