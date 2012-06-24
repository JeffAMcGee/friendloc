import os
import os.path
import json

import mock

from base import gob
from base.models import User, Edges, Tweets
from base.tests import SimpleGobTest
from base.tests.models_tests import MockTwitterResource, MockGisgraphyResource


def _patch_twitter():
    return mock.patch("base.twitter.TwitterResource",MockTwitterResource)

def _patch_gisgraphy():
    return mock.patch("explore.sprawl.GisgraphyResource",MockGisgraphyResource)


class TestSprawlToContacts(SimpleGobTest):
    @classmethod
    def setUpClass(cls):
        CWD = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(CWD,'..','..','tests','geo_tweets.json')
        with open(path) as file:
            cls.geotweets = [json.loads(l) for l in file]

    def setUp(self):
        super(TestSprawlToContacts, self).setUp()

    def test_mloc_users(self):
        mock_load = lambda s,p,e: self.geotweets
        with mock.patch.object(gob.Source,'load_output',mock_load):
            self.gob.run_job('mloc_users')
        results = self.FS['mloc_users.39']
        self.assertEqual(results[0]['id'], 51839)
        self.assertEqual(results[0]['mloc'], [-95.3,29.2])
        self.assertEqual(len(results), 1)

    def _find_edges_6(self):
        self.FS['mloc_users.06'] = [dict(id=6)]
        with _patch_twitter():
            self.gob.run_job('find_edges')

    def test_find_edges(self):
        self._find_edges_6()
        results = self.FS['find_edges.06']
        s_res = sorted(list(r[1])[0] for r in results)
        self.assertEqual(s_res, [0,1,2,3,7,12,18,24,30])
        flor = User.get_id(6)
        self.assertEqual(flor.just_mentioned,[7])
        self.assertEqual(sorted(flor.just_friends),[12,18,24,30])

    def test_find_edges_errors(self):
        self.FS['mloc_users.04'] = [dict(id=404)]
        self.FS['mloc_users.03'] = [dict(id=503)]
        with _patch_twitter():
            self.gob.run_job('find_edges')
        for uid in (404,503):
            missing = User.get_id(uid)
            self.assertEqual(missing.error_status,uid)
            self.assertEqual(missing.neighbors, None)
            self.assertEqual(missing.rfriends, None)
            self.assertEqual(Edges.get_id(uid), None)
            self.assertEqual(Tweets.get_id(uid), None)

    def test_contact_split(self):
        User(_id=24).save()
        self._find_edges_6()
        self.gob.run_job('contact_split')
        self.assertEqual(self.FS['contact_split.18'],[18])
        self.assertNotIn('contact_split.24', self.FS)

    def _fake_find_edges(self, *ids):
        self.FS['find_edges'] = [
                (User.mod_id(uid),[uid])
                for uid in ids
                ]

    def _lookup_contacts(self):
        with _patch_twitter():
            with _patch_gisgraphy():
                self.gob.run_job('contact_split')
                self.gob.run_job('lookup_contacts')

    def test_lookup_contacts(self):
        self._fake_find_edges(2,3)
        self._lookup_contacts()
        beryl = User.get_id(2)
        self.assertEqual(beryl.screen_name,'user_2')
        self.assertEqual(beryl.geonames_place.feature_code,'PPLA2')
        self.assertEqual(beryl.geonames_place.mdist,3)

    def test_pick_nebrs(self):
        self._fake_find_edges(2,3,7)
        self._lookup_contacts()
        flor = User(_id=6, just_mentioned=[7], just_friends=[1,2,3])
        flor.save()
