from base.tests import SimpleGobTest
from base.utils import use_mongo


class TestPrep(SimpleGobTest):
    def setUp(self):
        super(TestPrep,self).setUp()
        use_mongo('fl_fixture')

    def test_edge_d(self):
        self.FS['mloc_uids.03'] = [3]
        self.gob.run_job('training_users')
        self.gob.run_job('edge_d')
        edge03 = self.FS['edge_d.03']
        self.assertEqual(len(edge03),1)
        rels = edge03[0]['rels']
        self.assertEqual( [d['_id'] for d in rels], [1,6,9,2] )

    def test_nebrs_d(self):
        self.FS['mloc_uids.03'] = [3]
        self.gob.run_job('training_users')
        self.gob.run_job('nebrs_d')
        edge03 = self.FS['nebrs_d.03']
        self.assertEqual(len(edge03),1)
        nebrs = edge03[0]['nebrs']
        self.assertEqual( sorted(d['_id'] for d in nebrs), [1,2,6,9] )
