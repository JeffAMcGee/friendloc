from friendloc.base.tests import SimpleGobTest
from friendloc.base.utils import use_mongo
from friendloc.predict import fl


class TestFriendLoc(SimpleGobTest):
    def setUp(self):
        super(TestFriendLoc,self).setUp()
        use_mongo('fl_fixture')

    def test_edge_vect_flow(self):
        # integration test
        for x in xrange(100):
            self.FS['geo_ated.%02d'%x] = []
            self.FS['cheap_locals.%02d'%x] = []

        self.FS['mloc_uids.03'] = [3]
        self.FS['mloc_uids.06'] = [6]
        self.FS['mloc_blur'] = [1,[10],[2,4]]
        self.gob.run_job('pred_users')
        self.gob.run_job('nebrs_d')
        self.gob.run_job('nebr_vect')
        vects03 = self.FS['nebr_vect.03']
        # there are 4 relationships
        self.assertEqual(len(vects03),4)
