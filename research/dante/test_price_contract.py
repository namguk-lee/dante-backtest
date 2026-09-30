import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import pandas as pd
from collect_marcap_full_market import last_closed_date,append_verified_krx,append_recent_naver,assert_complete_recent_coverage

class PriceContractTests(unittest.TestCase):
    def fixture(self):
        x=pd.DataFrame([{'Date':pd.Timestamp('2026-09-28'),'Code':'041190','Name':'fixture','Market':'KOSDAQ','Close':5800}])
        q=pd.DataFrame([{'date':'2026-09-29','code':'041190','open':5760,'high':5780,'low':5670,'close':5750,'volume':776967,'amount':4445000000,'price_venue':'KRX','bar_status':'CLOSED','source_url':'https://stock.mk.co.kr/a','verification_url':'https://kind.krx.co.kr/a'}])
        return x,q
    def apply(self,q):
        x,_=self.fixture()
        with TemporaryDirectory() as d:
            p=Path(d)/'overlay.csv';q.to_csv(p,index=False)
            return append_verified_krx(x,p,pd.Timestamp('2026-09-29'),now='2026-09-30T12:22:00+09:00')
    def test_seoul_intraday_cutoff(self):
        self.assertEqual(last_closed_date('2026-09-30T03:22:00+00:00'),pd.Timestamp('2026-09-29'))
        self.assertEqual(last_closed_date('2026-09-30T15:40:00+09:00'),pd.Timestamp('2026-09-30'))
        self.assertEqual(last_closed_date('2026-10-03T12:00:00+09:00'),pd.Timestamp('2026-10-02'))
    def test_ambiguous_feed_blocked_even_when_venue_argument_would_be_ignored(self):
        with self.assertRaisesRegex(RuntimeError,'venue-specific'):append_recent_naver(pd.DataFrame(),pd.Timestamp('2026-09-29'))
    def test_reviewed_prices_and_amount_retained(self):
        _,q=self.fixture();r=self.apply(q).iloc[-1]
        self.assertEqual(r.Close,5750);self.assertEqual(r.Amount,4445000000)
        self.assertEqual(r.verification_status,'REVIEWED_OVERLAY')
        self.assertTrue(pd.isna(r.reported_change_pct))
    def test_invalid_contracts_rejected(self):
        for col,value in [('price_venue','NXT'),('bar_status','INTRADAY'),('date','2026-09-30'),('high',5700),('volume',0),('amount',float('nan')),('verification_url','https://stock.mk.co.kr/b'),('code','999999')]:
            with self.subTest(col=col):
                _,q=self.fixture();q[col]=value
                with self.assertRaises(ValueError):self.apply(q)
    def test_duplicates_rejected(self):
        _,q=self.fixture()
        with self.assertRaisesRegex(ValueError,'duplicate'):self.apply(pd.concat([q,q]))
    def test_reference_history_not_overwritten(self):
        _,q=self.fixture();q['date']='2026-09-28'
        with self.assertRaisesRegex(ValueError,'newer'):self.apply(q)
    def test_partial_overlay_cannot_certify_full_market_freshness(self):
        p=pd.DataFrame({'date':pd.to_datetime(['2026-09-28']*2+['2026-09-29']),
                        'code':['000001','000002','000001'],
                        'verification_status':['REFERENCE_HISTORY']*2+['REVIEWED_OVERLAY']})
        with self.assertRaisesRegex(RuntimeError,'1/2'):assert_complete_recent_coverage(p)
        more=p.iloc[-1:].copy();more['code']='000002'
        assert_complete_recent_coverage(pd.concat([p,more]))

if __name__=='__main__':unittest.main()
