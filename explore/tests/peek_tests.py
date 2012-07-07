from maroon import Model

import mock

from explore import peek
from base import models, utils
from base.tests import SimpleGobTest


class TestPeek(SimpleGobTest):
    def setUp(self):
        super(TestPeek,self).setUp()
        # FIXME: this is stupid boilerplate
        utils.use_mongo('fl_fixture')

    def test_contact_blur(self):
        saver = mock.patch.object(models.User,'save',mocksignature=True)
        with saver as s:
            list(peek.contact_blur(6))
            user = s.call_args[0][0]
        self.assertEqual(user.local_followers, .25)
        self.assertEqual(user.local_friends, 0)

    def test_edges_d(self):
        user_d = models.User.get_id(6).to_d()
        edges_d = peek.edges_d(user_d)
        self.assertEqual(len(edges_d),1)
        edge = edges_d[0]
        self.assertEqual(edge['jat']['_id'],7)
        self.assertEqual(edge['jat']['prot'],True)
        self.assertEqual(edge['rfrd']['_id'],0)
        self.assertNotIn('jfrd', edge)

    def test_edge_dists(self):
        jat = dict(ated=True, prot=False, mdist=1, lat=30, lng=-96)
        edge_d = dict(jat=jat, mloc=[-96,31])
        dists = list(peek.edge_dists(edge_d))
        self.assertEqual(len(dists),1)
        self.assertEqual(dists[0][0],('jat',True,False))
        self.assertAlmostEqual(dists[0][1],69.0976,places=3)


