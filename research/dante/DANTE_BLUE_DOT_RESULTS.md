# Dante Blue-dot / 파란점선 proxy study

Status: research-only. The proprietary formula is not public.

## Public evidence

Official/public recaps frequently describe price as touching or approaching the Blue-dot line. Official Q&A also indicates that versions can share a formula while using different numeric values, and that values may be changed depending on situation. Therefore a single fixed public constant set should not be called the exact formula.

## Dated explicit/near-touch labels

The current study contains six explicit/near-touch public examples; five matched the historical market panel directly and one required a missing-history fallback that was not available in the market-control pass.

## Internet hypotheses tested

- Bollinger upper 35 / 2, no shift
- Bollinger upper 20 / 2, shift 26

Additional grid families were tested only as sensitivity analysis. Grid winners are **not** adopted because selecting them after seeing a tiny labeled sample is overfitting.

## Result

### BB35 x 2, no shift

On the six labeled touch/near-touch examples in the parameter study:
- median same-day candle distance to the candidate line: ~0.39%
- 5/6 were within 1% on the analysis day
- 6/6 touched or came within the tested proximity during the latest five sessions

For the five directly matched same-date market-control cases, the labeled examples' exact-distance percentile ranged from about 4.7% to 21.8% (lower means fewer market stocks were at least as close).

Same-date whole-market frequency of being within 1% of BB35x2 varied materially by date, roughly 8.7% to 33.7%. Thus the proxy is moderately selective on some dates but not rare enough to use as a standalone signal.

### BB20 x 2, shift 26

This internet hypothesis fit materially worse:
- only 1/6 within 1% on the analysis day
- 4/6 within 3%
- weaker same-date labeled-vs-market separation.

## Overfitting check

A grid candidate around period 25 / multiplier 2.25 / shift 5 scored slightly better on the tiny positive sample, but this was found by parameter search after observing those examples. It is therefore treated as a mined fit and **not promoted**.

## Decision

Keep only a low-confidence research proxy:

`research_blue_dot_bb35_proxy = distance(candle, BollingerUpper(35, 2.0)) <= 1%`

Optional context field:

`research_blue_dot_bb35_recent5 = min distance over last 5 sessions <= 1%`

These names must retain `research_` / `proxy` wording. Do not label them as the proprietary Blue-dot indicator.

## Use in future matrix

The proxy may contribute to confluence only when combined with documented public structure such as:
- 256 / MA recovery
- MA-hit 112->224 or 224->448
- reverse-array recovery
- anchor/accumulation followed by quieter pullback
- concrete/support conversion
- research share 1:1 proxy

It should never be a standalone buy rule.