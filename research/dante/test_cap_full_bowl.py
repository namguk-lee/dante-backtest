#!/usr/bin/env python3
"""Small non-market unit tests for independent full-Gongguri gate comparison."""
import unittest
import pandas as pd
import numpy as np
from dante_cap_full_bowl_daily import is_full_bowl
from dante_cap_bowl_historical import outcome_summary
from dante_base_v4_daily_backtest import base_ok, trade_path
from cap_liquidity import attach_liquidity_features

class FullBowlTests(unittest.TestCase):
    def setUp(self):
        self.b={
            "bowl2_days":90, "bowl1_days":40,
            "bowl1_decline_pct":-.25,"below224_days_before_approach":95,
            "bowl2_core_range_pct":.20,"bowl2_core_net_change_pct":.05
        }
    def test_b90_full(self):
        self.assertTrue(is_full_bowl(self.b))
    def test_short_or_inverted_bowl_invalid(self):
        for patch in (
          {"bowl2_days":55},{"bowl1_days":110},{"bowl1_decline_pct":-.13},
          {"below224_days_before_approach":72},{"bowl2_core_range_pct":.51},
          {"bowl2_core_net_change_pct":.24}):
            x={**self.b,**patch}
            self.assertFalse(is_full_bowl(x),x)
    def test_completed_base_does_not_use_signal_bar(self):
        a=np.full(90,100,dtype=float)
        a[88]=102.
        self.assertTrue(base_ok(a,0,90)[0])
        # Completion of base excludes the event-day candle.
        b=np.r_[a,185.]
        self.assertTrue(base_ok(b,0,90)[0])
    def test_cap_can_pass_but_legacy_fail(self):
        n=70
        g=pd.DataFrame({
          "close":np.full(n,10000.),
          "volume":np.full(n,20000.),
          "amount":np.full(n,200e6),
          "marcap":np.full(n,50e9),
          "listed_shares":np.full(n,5e6)
        })
        z=attach_liquidity_features(g)
        self.assertTrue(bool(z.adaptive_liquidity_pass.iloc[-1]))
        self.assertFalse(bool(z.legacy_50e_pass.iloc[-1]))
    def test_maturity_denominator_not_timeout(self):
        e=pd.DataFrame([
          dict(base_days=60,setup="EMA224_BREAKOUT",year=2020,gate="LEGACY_50E",
               code="A",net_20d=4.2,hold_net_20d=3.1,outcome_20d="TARGET",
               cap_before_event_won=50e9,pre20_amount_won=250e6),
          dict(base_days=60,setup="EMA224_BREAKOUT",year=2020,gate="LEGACY_50E",
               code="B",net_20d=np.nan,hold_net_20d=np.nan,outcome_20d=None,
               cap_before_event_won=50e9,pre20_amount_won=250e6)
        ])
        s=outcome_summary(e).iloc[0]
        self.assertEqual(int(s.events),2)
        self.assertEqual(int(s.mature20),1)
        self.assertEqual(int(s.immature20),1)
        self.assertEqual(int(s.target20),1)
        self.assertEqual(int(s.timeout20),0)
    def test_2r_gap_stop_model(self):
        op=np.array([100.,85.,100.,100.])
        hi=np.array([100.,105.,105.,105.])
        lo=np.array([100.,80.,90.,90.])
        cl=np.array([100.,90.,101.,101.])
        x=trade_path(op,hi,lo,cl,0,100,90,2)
        self.assertEqual(x["outcome"],"STOP_GAP")
        self.assertAlmostEqual(x["net"],-15.4,places=5)

if __name__=="__main__":
    unittest.main()
