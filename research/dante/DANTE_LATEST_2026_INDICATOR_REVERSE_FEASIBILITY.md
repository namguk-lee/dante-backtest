# Dante latest 2026 indicator reverse-calculation feasibility (2026-11-14)

## Scope and source order
1. 2026-11-01 official Finup packages: blue dotted color variant (swing position), Watermelon (volatile markets), Rainbow/Bridge (medium/long trend), with official paid construction videos. **Exact math inaccessible from the public product summary.** https://stock.finup.co.kr/paid/search/154849/board
2. 2026-11-01 official Finup course and scanner bundle: Bottom Odori, Odori (Dante), Reverse-Array Odori2; wave projection/target-setting. https://stock.finup.co.kr/paid/lecture/154842
3. Official Finup rebound signal: 5-day moving average turns; publisher marketing describes 22 detections per month, not audited performance or our scan. https://stock.finup.co.kr/paid/robo-pack/154097
4. Official Finup accumulation signal: inverse MA arrangement -> accumulation candle -> months of flat/concrete -> blue-dotted breakout; publisher describes 41 per month (not audited performance). https://stock.finup.co.kr/paid/robo-pack/154032
5. Official 2022 Q&A: blue dotted V1/V2 use same formula with different parameters; upgraded indicator forms a band, original uses dots; parameters modified depending on circumstances. https://m.xn--6j1bp61aksejsj.com/article/강의-qna/6/30017/
6. Third-party gold/silver ParaSAR hypothesis, **not officially attested as exact settings**: default step .02/max .2, blog alternatives .009/.018 and .007/.15. https://valueupstory.tistory.com/entry/주식단테-금밭은밭-강세약세적용-Para-SAR-설정-후기HPSP-적용-키움증권-영웅문-hts

## Identifiability tiers
- Exact standard mathematical indicators: EMA(n), SMA(n), standard Parabolic SAR once both parameters **and calculation convention** are known, BB(n,k) once periods/dev/price basis known, standard AVWAP with anchor and volume known. **These are standard constructs, not proof of exact paid Dante code.**
- Observable-condition approximation: '5-day MA turning', price reclaim/settle around EMA112/224/448, 256 EMA5/112/224 medium-term, accumulation/long base/blue-line breakout sequence. Need distinguish SMA5 from EMA5, multi-bar confirmation, support rules.
- Unidentified proprietary internals: precise blue-dotted level/phase/color variants, Watermelon event trigger, Rainbow/Bridge ribbon and color state, reverse-MA proprietary arrow. Label all independent approximations PROXY, never exact product.
- Avoid inferring a paid formula from a single screenshot/price chart, because many parameters fit one example; require dated positives and negatives and line labels for calibrated validation.

## Small same-data proof of parameter non-identifiability (not accuracy)
Study input: the 12 **preselected** Nov 2026 Korean market stocks' real daily OHLCV, only five sessions Nov 9–13. This is a biased illustrative panel, not a market scan.
- EMA5 turned-up events (with close > EMA5): 8
- SMA5 turned-up events (with close > SMA5): 6
- Standard Parabolic SAR bullish flips step .02/max .2: 6
- Slow third-party blog hypothetical ParaSAR step .009/max .018 bullish flips: 2
- Candles near BB upper (20,2): 5; BB upper (35,2): 5; BB upper (40,2): 6
**No true output labels from paid Dante indicators were present. Therefore NONE of these counts is a match rate, precision, recall, or reverse-engineering success.**

## Fit-and-validate protocol
1. Gather 30–50 publicly visible dated positive illustrations plus independent negatives across 10+ symbols, with output labels and actual pre-signal OHLCV. Screenshots showing paid indicator line price-scale values are better than signal dates alone; licensed material only when voluntarily provided by its holder.
2. For price-line blue dot: candidate families BB mid + k*std, shifted BB, SMA/EMA envelopes, ATR channels. Sweep period, multiplier, shift using ONLY training dates, evaluate median relative price-line error AND same-day signal F1/false positive with locked thresholds on withheld dates/stocks. Test whether historical screenshots repaint.
3. For binary Watermelon/Rainbow: identify positive signal dates and negatives, fit publicly computable price/volume/volatility/EMA families and **calibrate** under no-look-ahead constraints. Without output labels, this model cannot be meaningfully reverse identified.
4. Compare independent stage candidates to current A/B prior workflow and track false-positive rate, weekly signal counts, entry next open, MAE/MFE and cost-adjusted returns. Correct for delisting and survivorship bias.
5. Use prospectively unobserved dates for forward confirmation; prior 2018–2026 tuning is not clean out-of-sample.

No automated trades, no modifications to the mainline production scanner.
