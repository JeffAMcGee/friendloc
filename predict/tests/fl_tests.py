import unittest

from base.utils import use_mongo
from base.gob import SimpleEnv, Gob
from predict import fl
from gb import create_jobs


class TestFriendLoc(unittest.TestCase):
    def setUp(self):
        self.gob = Gob(SimpleEnv())
        use_mongo('fl_fixture')
        create_jobs(self.gob)
        SimpleEnv.THE_FS = {}

    def test_edge_vect_flow(self):
        # integration test
        self.gob.run_job('mloc_users')
        self.gob.run_job('edges_d')
        self.gob.run_job('edge_vect')
        vects03 = SimpleEnv.THE_FS['edge_vect.03']
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
        # (dist, [priv, ated, is_frd, is_fol, mdist, folc]
        self.assertEqual(next(vect), (7, [0, 1, 1, 0, 2, 3]))


