"""Test per-date Gongguri row aggregation without touching production signals."""
import unittest
import tempfile
from pathlib import Path
import pandas as pd
from dante_full_bowl_from_classic_results import process,strict_full_bowl

class PostprocessTests(unittest.TestCase):
    def test_only_full_base_and_distinct_anchor_days(self):
        d=dict(date="2026-11-13",code="084695",name="대상홀딩스우",
               exchange="KO",bowl2_days=90,bowl1_days=35,
               bowl1_decline_pct=-30,bowl2_core_range_pct=18,
               bowl2_core_net_change_pct=1,
               below224_days_before_approach=90,
               anchor_date="2026-10-13",anchor_alive_strict=True,
               concrete_breakout=True,concrete_hold_loose=True,
               tier="NEAR",market_cap_quality="CAP_MISSING",
               cap_tier="CAP_UNKNOWN",anchor_liquidity_relative_pct=.25,
               anchor_volume_ratio=2.3,market_cap_won=float("nan"))
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)
            pd.DataFrame([d]).to_csv(f/"legacy.csv",index=False)
            alt={**d,"anchor_date":"2026-11-11"}
            pd.DataFrame([alt]).to_csv(f/"cap.csv",index=False)
            rows,s=process(f/"legacy.csv",f/"cap.csv")
            self.assertEqual(len(rows),2)
            self.assertEqual(set(rows.comparison),{"BOTH"})
            self.assertEqual(rows.anchor_date.nunique(),2)
            self.assertTrue(rows.preferred_share_check.all())
            self.assertTrue(rows.no_automatic_buy.all())
            self.assertTrue(rows.liquidity_and_price_warning.eq("CURRENT_CAP_UNAVAILABLE").all())
    def test_strict_no_short_base_or_short_ema224_history(self):
        x=pd.DataFrame([dict(bowl2_days=58,bowl1_days=30,
                 bowl1_decline_pct=-25,below224_days_before_approach=90,
                 bowl2_core_range_pct=10,bowl2_core_net_change_pct=2),
              dict(bowl2_days=100,bowl1_days=30,
                 bowl1_decline_pct=-25,below224_days_before_approach=75,
                 bowl2_core_range_pct=10,bowl2_core_net_change_pct=2)])
        self.assertEqual(int(strict_full_bowl(x).sum()),0)
    def test_new_anchor_not_implied_from_base_status(self):
        d=dict(date="2026-11-13",code="005930",name="삼성전자",
               exchange="KO",bowl2_days=90,bowl1_days=35,
               bowl1_decline_pct=-30,bowl2_core_range_pct=18,
               bowl2_core_net_change_pct=1,
               below224_days_before_approach=90,
               anchor_date="2026-10-13",anchor_alive_strict=False,
               concrete_breakout=False,concrete_hold_loose=False,
               tier="NEAR",market_cap_quality="CAP_MISSING",
               cap_tier="CAP_UNKNOWN",anchor_liquidity_relative_pct=.25,
               anchor_volume_ratio=1.9,market_cap_won=float("nan"))
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)
            pd.DataFrame([d]).to_csv(f/"old.csv",index=False)
            pd.DataFrame([d]).to_csv(f/"new.csv",index=False)
            x,s=process(f/"old.csv",f/"new.csv")
            self.assertTrue(x.stage.eq("ANCHOR_INVALIDATED").all())
            self.assertFalse(x.anchor_is_new_on_analysis_date.any())
            self.assertFalse(x.preferred_share_check.any())

if __name__=="__main__":
    unittest.main()
