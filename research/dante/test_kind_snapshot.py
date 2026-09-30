import unittest
from collect_kind_research_snapshot import parse_kind
import pandas as pd
from dante_snapshot_review import validate_recent_coverage


class KindSnapshotTests(unittest.TestCase):
    def fixture(self, code='041190', date='2026-09-30', time='15:30:20', high=5830):
        pairs=[('현재가',5740),('전일가',5750),('거래량(주)',670680),('시가',5720),('거래대금(원)',3849355205),('고가',high),('시가총액',458052000000),('저가',5670)]
        return f'<input id="repIsuSrtCd" value="A{code}"><p>{date} {time} 기준</p><table class="detail">'+''.join(f'<tr><th>{k}</th><td>{v:,}</td></tr>' for k,v in pairs)+'</table>'

    def test_parse_prices_without_confusing_market_cap_with_open(self):
        r=parse_kind(self.fixture(),'041190','2026-09-30')
        self.assertEqual(r['open'],5720);self.assertEqual(r['close'],5740)
        self.assertEqual(r['amount'],3849355205)

    def test_preferred_share_must_not_use_ordinary_issuer_quote(self):
        with self.assertRaisesRegex(ValueError,'mismatch'):
            parse_kind(self.fixture(),'041195','2026-09-30')

    def test_stale_and_intraday_snapshots_rejected(self):
        for args in ({'date':'2026-09-29'},{'time':'15:20:00'}):
            with self.subTest(args=args),self.assertRaisesRegex(ValueError,'stale or intraday'):
                parse_kind(self.fixture(**args),'041190','2026-09-30')

    def test_invalid_high_rejected(self):
        with self.assertRaisesRegex(ValueError,'envelope'):
            parse_kind(self.fixture(high=5700),'041190','2026-09-30')

    def test_partial_or_mixed_recent_file_cannot_match_full_collection(self):
        ledger=pd.DataFrame({'code':['041190','234340'],'status':['OK_RESEARCH_ONLY']*2})
        recent=pd.DataFrame({'code':['041190']*2,'date':['2026-09-29','2026-09-30']})
        with self.assertRaisesRegex(ValueError,'coverage disagree'):validate_recent_coverage(recent,ledger)
        ledger=ledger.iloc[:1]
        with self.assertRaisesRegex(ValueError,'incomplete recent dates'):validate_recent_coverage(recent.iloc[:1],ledger)
        with self.assertRaisesRegex(ValueError,'duplicate'):validate_recent_coverage(pd.concat([recent,recent]),ledger)


if __name__=='__main__':unittest.main()
