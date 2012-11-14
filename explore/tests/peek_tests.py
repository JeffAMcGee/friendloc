import math
import mock

from explore import peek
from base import models, utils
from base.tests import SimpleGobTest


class TestPeek(SimpleGobTest):
    def setUp(self):
        super(TestPeek,self).setUp()
        # FIXME: this is stupid boilerplate
        utils.use_mongo('fl_fixture')

    def _assert_stranger_prob(self,iter,expected_spot,expected_dist):
        spot,prob = next(iter)
        self.assertEqual(spot,expected_spot)
        dist = 1/(1-math.e**prob)
        self.assertAlmostEqual(dist,expected_dist,places=2)

    def test_stranger_probs(self):
        contact_count = {(-1799,600):1}.iteritems()
        with mock.patch.object(utils,'contact_prob',lambda m:1.0/m):
            mat = peek._contact_mat(contact_count)
            row = peek.stranger_prob(600,mat)
            self._assert_stranger_prob(row,(-1800,600),3.45)
            self._assert_stranger_prob(row,(-1799,600),2)
            above = peek.stranger_prob(610,mat)
            self._assert_stranger_prob(above,(-1800,610),69.18)

    def test_stranger_dists(self):
        self.FS['mloc_uids.03'] = [3]
        for contact in xrange(4):
            self.FS['contact_split.%02d'%contact] = [contact]

        self.gob.run_job('contact_count')
        self.gob.run_job('mloc_tile')
        self.gob.run_job('tile_split')
        self.gob.run_job('stranger_dists')
        results = self.FS['stranger_dists.300']

        dists_,counts = zip(*results)
        dists = sorted(int(d) for d in dists_)
        self.assertEqual(counts,(1,1,1,1))
        self.assertEqual(dists,[23,24,27,31])

    def test_nebr_dists(self):
        self.FS['mloc_uids.03'] = [3]
        self.gob.run_job('mloc_tile')
        self.gob.run_job('tile_split')
        self.gob.run_job('nebr_dists')
        results = self.FS['nebr_dists.300']

        dists_,counts = zip(*results)
        dists = sorted(int(d) for d in dists_)
        self.assertEqual(counts,(1,1,1,1))
        self.assertEqual(dists,[21,24,45,65])

    def test_contact_count(self):
        self.FS['contact_split.04'] = [4]
        self.FS['contact_split.05'] = [5,6]
        self.gob.run_job('contact_count')
        four = self.FS['contact_count.04']
        self.assertEqual(four, [((-964, 304), 1)])
        sums = self.FS['contact_count']
        self.assertEqual(dict(sums),{(-964,306):1,(-964,305):1,(-964,304):1})

    def test_contact_blur(self):
        saver = mock.patch.object(models.User,'save',mocksignature=True)
        with saver as s:
            list(peek.contact_blur(6))
            user = s.call_args[0][0]
        self.assertEqual(user.local_followers, .25)
        self.assertEqual(user.local_friends, 0)

    def test_edges_d(self):
        for x in xrange(100):
            self.FS['geo_ats.%02d'%x] = []
        instance = peek.EdgesDict(self.env)
        user_d = models.User.get_id(6).to_d()
        edges_d = instance.edges_d(user_d)
        self.assertEqual(len(edges_d),1)
        edge = edges_d[0]
        self.assertEqual(edge['jat']['_id'],7)
        self.assertEqual(edge['jat']['prot'],True)
        self.assertEqual(edge['rfrd']['_id'],0)
        self.assertNotIn('jfrd', edge)

    def test_edge_dists(self):
        jat = dict(i_at=True, u_at=True, prot=False, mdist=1, lat=30, lng=-96)
        edge_d = dict(jat=jat, mloc=[-96,31])
        dists = list(peek.edge_dists(edge_d))
        self.assertEqual(len(dists),1)
        self.assertEqual(dists[0][0],('jat',True,True,False))
        self.assertAlmostEqual(dists[0][1],69.0976,places=3)


