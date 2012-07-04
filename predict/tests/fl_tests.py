from base.tests import SimpleGobTest
from base.utils import use_mongo
from predict import fl


class TestFriendLoc(SimpleGobTest):
    def setUp(self):
        super(TestFriendLoc,self).setUp()
        use_mongo('fl_fixture')

    def test_edge_vect_flow(self):
        # integration test
        self.FS['mloc_uids.03'] = [3]
        self.FS['mloc_uids.06'] = [6]
        self.gob.run_job('training_users')
        self.gob.run_job('edge_d')
        self.gob.run_job('edge_vect')
        vects03 = self.FS['edge_vect.03']
        # there are 4 relationships
        self.assertEqual(len(vects03),4)

    def test_edge_vect(self):
        rel = dict(lat=31, lng=-96, kind=6, folc=7, mdist=3)
        user = dict(
                rels = [rel],
                _id = 3,
                mloc = [-96,30],
                )
        vect = fl.edge_vect(user)
        # (priv, ated, is_frd, is_fol, mdist, folc, dist)
        self.assertEqual(next(vect), [0, 1, 1, 0, 2, 3, 8, 7])


