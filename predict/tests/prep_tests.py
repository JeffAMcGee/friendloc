import unittest

from base.utils import use_mongo
from base.gob import SimpleEnv, Gob
from gb import create_jobs


class TestPrep(unittest.TestCase):
    def setUp(self):
        self.gob = Gob(SimpleEnv())
        use_mongo('fl_fixture')
        create_jobs(self.gob)
        SimpleEnv.THE_FS = {}

    def test_mloc_users(self):
        self.gob.run_job('mloc_users')
        users03 = SimpleEnv.THE_FS['mloc_users.03']
        self.assertEqual(len(users03),1)
        self.assertEqual(users03[0]['name'], 'Chris')
        self.assertEqual(users03[0]['folc'], 9)

    def test_edges_d(self):
        self.gob.run_job('mloc_users')
        self.gob.run_job('edges_d')
        edges03 = SimpleEnv.THE_FS['edges_d.03']
        self.assertEqual(len(edges03),1)
        rels = edges03[0]['rels']
        self.assertEqual( [d['_id'] for d in rels], [1,6,9,2] )
