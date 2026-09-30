"""Regression tests for causal support and overlapping recovery labels."""
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np
import pandas as pd
import dante_signal_matrix_research as matrix


def support_path(break_i=110, retest_i=None, breach_i=None):
    n=120
    high=np.linspace(80,99,n)
    close=high-2
    low=high-3
    high[50]=110  # Unique confirmed resistance pivot.
    for i in range(break_i,n):
        close[i]=116; high[i]=118; low[i]=115
    close[break_i]=112; high[break_i]=114
    low[break_i]=109
    if retest_i is not None:
        close[retest_i]=111; high[retest_i]=113; low[retest_i]=110
    if breach_i is not None:
        close[breach_i]=105; high[breach_i]=106; low[breach_i]=104
    return pd.DataFrame({'date':pd.bdate_range('2025-01-01',periods=n),
                         'ah':high,'al':low,'ac':close})


class ConcreteTests(unittest.TestCase):
    def test_breakout_candle_is_not_a_retest(self):
        self.assertFalse(matrix.concrete_proxy(support_path(break_i=119))[0])

    def test_later_retest_is_required(self):
        self.assertFalse(matrix.concrete_proxy(support_path())[0])

    def test_later_retest_is_accepted(self):
        z=support_path(retest_i=112)
        match,level,broken,retest=matrix.concrete_proxy(z)
        self.assertTrue(match)
        self.assertEqual(level,110)
        self.assertEqual(broken,z.at[110,'date'])
        self.assertEqual(retest,z.at[112,'date'])
        self.assertGreater(retest,broken)

    def test_breach_before_retest_invalidates_cycle(self):
        self.assertFalse(matrix.concrete_proxy(support_path(retest_i=114,breach_i=112))[0])

    def test_breach_after_retest_stays_invalid_after_recovery(self):
        self.assertFalse(matrix.concrete_proxy(support_path(retest_i=112,breach_i=114))[0])


class ConfluenceTests(unittest.TestCase):
    def test_overlapping_112_detectors_count_once(self):
        n=520
        z=pd.DataFrame({'date':pd.bdate_range('2022-01-01',periods=n),
            'code':'000001','name':'fixture','exchange':'KQ','close':105.,'ac':105.,
            'ah':106.,'al':104.,'ema5':103.,'ema112':100.,'ema224':110.,'ema448':120.,
            'amount20':6e9,'vr20':.9,'bb35_upper2':130.,
            'ema112_slope20':1.,'ema224_slope20':-1.,'ema448_slope20':-1.})
        with patch.object(matrix,'last_cross_days',return_value=3), \
             patch.object(matrix,'last_price_cross_days',return_value=3), \
             patch.object(matrix,'share_proxy_for_ma',return_value=None), \
             patch.object(matrix,'concrete_proxy',return_value=(False,np.nan,pd.NaT,pd.NaT)), \
             patch.object(matrix,'bowl3_structure_proxy',return_value={'match':False}):
            r=matrix.scan(z,5e9)
        self.assertTrue(r['public_256_long'])
        self.assertTrue(r['public_ma_hit_112_224'])
        self.assertEqual(r['public_technique_raw_count'],2)
        self.assertEqual(r['public_technique_count'],1)
        self.assertEqual(r['confluence_count'],1)
        self.assertEqual(r['public_technique_families'],'RECOVERY_112')
        self.assertEqual(r['entry_validation'],'NOT_VALIDATED')


class DatedCaseTests(unittest.TestCase):
    def test_stale_case_is_a_miss_not_a_match(self):
        from dante_official_case_audit import audit
        panel=pd.DataFrame({'date':[pd.Timestamp('2022-11-18')],
            'series_id':['000001_E01'],'name':['fixture']})
        labels=pd.DataFrame([{'analysis_date':pd.Timestamp('2025-11-17'),
                             'stock':'fixture'}])
        with TemporaryDirectory() as d:
            out=audit(panel,labels,Path(d))
            misses=pd.read_csv(Path(d)/'official_case_misses.csv')
        self.assertTrue(out.empty)
        self.assertEqual(misses.iloc[0].reason,'stale_history')

    def test_renamed_history_preserved_and_future_bars_excluded(self):
        from dante_official_case_audit import audit
        n=550
        prices=np.linspace(100,120,n)
        panel=pd.DataFrame({'date':pd.bdate_range('2022-01-01',periods=n),
            'series_id':'000001_E01','code':'000001','exchange':'KQ',
            'name':['old name']*490+['new name']*60,
            'open':prices,'high':prices+1,'low':prices-1,
            'close':prices,'adjusted_close':prices,'amount':6e9,'volume':1e6})
        dt=panel.at[499,'date']
        # Extreme future prices must not enter analysis-day MAs or chart data.
        panel.loc[500:,'adjusted_close']=1e9
        labels=pd.DataFrame([{'analysis_date':dt,'stock':'new name',
            'techniques':'test','context_tags':'test','vip_indicators':'',
            'outcome':'unknown','stop_hit':False,'source_url':'https://example.com'}])
        with TemporaryDirectory() as d, patch('dante_official_case_audit.plot_case') as plot:
            out=audit(panel,labels,Path(d))
        self.assertEqual(out.iloc[0].history_rows,500)
        self.assertEqual(out.iloc[0].market_date,dt)
        chart_z=plot.call_args.args[0]
        self.assertLess(chart_z.ac.max(),121)
        self.assertEqual(chart_z.date.max(),dt)


class MarketUniverseTests(unittest.TestCase):
    def test_kosdaq_global_is_retained_in_kosdaq(self):
        from collect_marcap_full_market import load_year
        x=pd.DataFrame({'Date':pd.to_datetime(['2025-01-02']*3),
            'Code':['091700','000001','000002'], 'Name':['fixture']*3,
            'Market':['KOSDAQ GLOBAL','KOSPI','KONEX'],
            'Open':[100]*3,'High':[101]*3,'Low':[99]*3,'Close':[100]*3,
            'Volume':[1000]*3,'Amount':[100000]*3})
        with TemporaryDirectory() as d:
            path=Path(d)/'year.parquet';x.to_parquet(path,index=False)
            got=load_year(path,pd.Timestamp('2025-01-01'),pd.Timestamp('2025-01-03'))
        self.assertEqual(set(got.Code),{'091700','000001'})
        self.assertEqual(got.loc[got.Code.eq('091700'),'Market'].iloc[0],'KOSDAQ')


if __name__=='__main__':
    unittest.main()
