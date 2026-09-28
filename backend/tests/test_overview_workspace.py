import unittest
from urllib.parse import urlsplit

from backend.data import Repository
from backend.overview import overview_data, work_status, link
from frontend.navigation import navigation_context


class OverviewWorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = Repository()
        cls.data = overview_data(cls.repo)
        cls.meta = cls.repo.meta()

    def test_cards_and_work_items_use_current_full_scope(self):
        data = self.data
        demand = self.repo.demand()
        maintenance = self.repo.maintenance()
        rows,counts = work_status(data,{})
        self.assertEqual(counts['demand']['total'],demand['reviewCount'])
        self.assertEqual(counts['maintenance']['total'],len(maintenance['events']))
        self.assertEqual(data['maintenance']['alertCycles'],maintenance['alertCycles'])
        self.assertEqual(data['demand']['reviewCount'],data['demand']['upwardReviewCount']+data['demand']['downwardReviewCount'])
        self.assertEqual(len(rows),demand['reviewCount']+len(maintenance['events'])+1)
        self.assertEqual(len({r['key'] for r in rows}),len(rows))

    def test_unavailable_is_not_zero_or_unconfirmed(self):
        rows,counts = work_status(self.data,{'__unavailable__':True})
        self.assertTrue(all(r['status']=='조회 불가' and r['done'] is None for r in rows))
        self.assertTrue(all(c['remaining'] is None and c['done'] is None for c in counts.values()))

    def test_exact_ids_exclude_old_models_other_dates_and_old_versions(self):
        rows,_ = work_status(self.data,{})
        demand = next(r for r in rows if r['track']=='demand')
        maintenance = next(r for r in rows if r['track']=='maintenance')
        saved = {'decision':'reviewed','dataVersion':self.data['dataVersion']}
        records = {demand['key']:saved,maintenance['key']:saved,maintenance['key'].replace('maintenance-alarm-v3','old-policy'):saved,
                   demand['key'].replace(self.data['demand']['targetDate'],'1900-01-01'):saved}
        actual,counts = work_status(self.data,records)
        self.assertEqual(counts['demand']['done'],1)
        self.assertEqual(counts['maintenance']['done'],1)
        records[demand['key']] = {**saved,'dataVersion':'old-seed'}
        actual,counts = work_status(self.data,records)
        self.assertEqual(counts['demand']['done'],0)
        self.assertEqual(next(r for r in actual if r['key']==demand['key'])['status'],'재확인 필요')

    def test_quality_human_decisions_are_separate_from_ai_and_completion(self):
        key = 'quality:'+self.data['quality']['testId']
        for decision,status,remaining in [('clear','이상 없음',0),('hold','출하 보류',1),('retest','재시험 요청',1)]:
            rows,counts = work_status(self.data,{key:{'decision':decision,'dataVersion':self.data['dataVersion']}})
            self.assertEqual(next(r for r in rows if r['key']==key)['status'],status)
            self.assertEqual(counts['quality']['remaining'],remaining)
            self.assertGreater(self.data['quality']['abnormalSegmentCount'],0)

    def test_every_link_restores_exact_context(self):
        for row in self.data['workItems']:
            parsed = urlsplit(row['route'])
            context = navigation_context(row['track'],'?'+parsed.query,self.meta)
            self.assertEqual(context['tab'],'review')
            self.assertTrue(context['resetLocalFilters'])
            if row['track']=='demand':
                self.assertEqual(context['d-part'],row['target'])
                self.assertEqual(context['d-date'],self.data['demand']['targetDate'])
                self.assertEqual(context['d-model'],self.data['demand']['operatingModel'])
            elif row['track']=='maintenance':
                self.assertEqual(context['event'],row['eventId'])
                self.assertEqual(context['m-run'],self.data['maintenance']['sourceFile'])
                self.assertEqual(context['m-sup'],self.meta['defaultSupervised'])
            else:
                self.assertEqual(context['q-test'],row['target'])
                self.assertEqual(context['q-cell'],self.data['quality']['suspectedCell'])

    def test_invalid_url_values_are_not_applied(self):
        context = navigation_context('demand','?tab=review&part=nonexistent&date=1900-01-01&model=evil',self.meta)
        self.assertNotIn('d-part',context)
        self.assertNotIn('d-date',context)
        self.assertNotIn('d-model',context)
        self.assertIsNone(navigation_context('maintenance','?event=anything',self.meta)['event'])
        self.assertNotIn('q-cell',navigation_context('quality','?cell=M17CV12',self.meta))


if __name__=='__main__':
    unittest.main()
