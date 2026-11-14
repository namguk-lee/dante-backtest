#!/usr/bin/env python3
"""Small deterministic tests for cap-aware liquidity; stdlib unittest."""
import unittest
import numpy as np
import pandas as pd
from cap_liquidity import attach_liquidity_features, CAP_TIERS, market_cap_tier


def sample(cap=50_000_000_000.,avg=250_000_000.,raw_unit="won",periods=45):
    close=np.full(periods,10000.)
    shares=np.full(periods,cap/10000.)
    amount=np.full(periods,avg)
    if raw_unit=="won":rawcap=np.full(periods,cap)
    elif raw_unit=="million":rawcap=np.full(periods,cap/1e6)
    else:rawcap=np.full(periods,np.nan)
    return pd.DataFrame({
        "date":pd.bdate_range("2026-01-05",periods=periods),
        "close":close,"volume":amount/10000.,"amount":amount,
        "listed_shares":shares,"marcap":rawcap,
    })

class MarketCapTests(unittest.TestCase):
    def test_won_and_million_both_normalize(self):
        for unit in ("won","million"):
            z=attach_liquidity_features(sample(raw_unit=unit))
            self.assertAlmostEqual(z.market_cap_won.iloc[-1],5e10)
            self.assertIn("CHECKED",z.market_cap_quality.iloc[-1])
    def test_small_company_passes_below_legacy_50e(self):
        z=attach_liquidity_features(sample())
        a=z.iloc[-1]
        self.assertEqual(a.cap_tier,"SMALL_LT_100B")
        self.assertTrue(bool(a.adaptive_liquidity_pass))
        self.assertFalse(bool(a.legacy_50e_pass))
        self.assertAlmostEqual(a.relative_turnover_pre20_pct,.5,places=4)
    def test_giant_fails_weak_relative_intensity(self):
        z=attach_liquidity_features(sample(cap=5e13,avg=1.5e9))
        self.assertFalse(bool(z.adaptive_liquidity_pass.iloc[-1]))
        self.assertEqual(z.cap_tier.iloc[-1],"LARGE_GE_1T")
    def test_event_amount_must_not_inflate_pre20_baseline(self):
        g=sample()
        before=attach_liquidity_features(g).iloc[-1]
        g.loc[len(g)-1,"amount"]=99e9
        after=attach_liquidity_features(g).iloc[-1]
        self.assertEqual(before.pre20_amount_won,after.pre20_amount_won)
        self.assertEqual(before.cap_before_event_won,after.cap_before_event_won)
        self.assertGreater(after.amount_spike_vs_prior20,before.amount_spike_vs_prior20)
    def test_no_future_capitalization(self):
        g=sample()
        reference=attach_liquidity_features(g).cap_before_event_won.iloc[-1]
        g.loc[len(g)-1,"marcap"]=5e13
        g.loc[len(g)-1,"listed_shares"]=5e9
        z=attach_liquidity_features(g)
        self.assertEqual(reference,z.cap_before_event_won.iloc[-1])
    def test_missing_shares_after_five_rows_not_silently_filled(self):
        g=sample()
        g.loc[30:,"listed_shares"]=np.nan
        g.loc[30:,"marcap"]=np.nan
        z=attach_liquidity_features(g)
        self.assertEqual(z.market_cap_quality.iloc[33],"LAST_REPORTED_SHARES_ESTIMATE")
        self.assertEqual(z.market_cap_quality.iloc[-1],"CAP_MISSING")
        self.assertFalse(bool(z.adaptive_liquidity_pass.iloc[-1]))
    def test_mismatched_raw_cap_never_silently_accepted(self):
        g=sample()
        g.loc[g.index[-1],"marcap"]=7e10
        z=attach_liquidity_features(g)
        self.assertTrue(bool(z.cap_reported_conflict.iloc[-1]))
        self.assertAlmostEqual(z.market_cap_won.iloc[-1],5e10)
        self.assertEqual(z.market_cap_quality.iloc[-1],"MARCAP_MISMATCH_ESTIMATE")
    def test_output_not_free_float(self):
        z=attach_liquidity_features(sample())
        self.assertIn("listed_share_volume_turnover_pct",z.columns)
        self.assertNotIn("free_float_turnover",z.columns)
    def test_tier_boundaries(self):
        self.assertEqual(market_cap_tier(99_999_999_999.),"SMALL_LT_100B")
        self.assertEqual(market_cap_tier(100_000_000_000.),"MID_100B_TO_1T")
        self.assertEqual(market_cap_tier(1_000_000_000_000.),"LARGE_GE_1T")
        self.assertEqual(market_cap_tier(float("nan")),"CAP_UNKNOWN")

if __name__=="__main__":
    unittest.main()
