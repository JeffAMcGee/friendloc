import os
import os.path
import json

import mock

from base import gob
from base.models import User, Edges, Tweets
from base.tests import SimpleGobTest
from base.tests.models_tests import MockTwitterResource, MockGisgraphyResource
from explore import sprawl


def _patch_twitter():
    return mock.patch("base.twitter.TwitterResource",MockTwitterResource)

def _patch_gisgraphy():
    return mock.patch("explore.sprawl.GisgraphyResource",MockGisgraphyResource)


class TestSprawl(SimpleGobTest):
    @classmethod
    def setUpClass(cls):
        CWD = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(CWD,'..','..','tests','geo_tweets.json')
        with open(path) as file:
            cls.geotweets = [json.loads(l) for l in file]

    def setUp(self):
        super(TestSprawl, self).setUp()

    def test_mloc_users(self):
        mock_load = lambda s,p,e: self.geotweets
        with mock.patch.object(gob.Source,'load_output',mock_load):
            self.gob.run_job('parse_geotweets')
        self.gob.run_job('mloc_users')
        results = self.FS['mloc_users.39']
        self.assertEqual(results[0]['id'], 51839)
        self.assertEqual(results[0]['mloc'], [-95.3,29.2])
        self.assertEqual(len(results), 1)

    def _find_contacts_6(self):
        self.FS['mloc_users.06'] = [dict(id=6)]
        with _patch_twitter():
            self.gob.run_job('find_contacts')

    def test_find_contacts(self):
        self._find_contacts_6()
        results = self.FS['find_contacts.06']
        s_res = sorted(list(r[1])[0] for r in results)
        self.assertEqual(s_res, [0,1,2,3,7,12,18,24,30])
        flor = User.get_id(6)
        self.assertEqual(flor.just_mentioned,[7])
        self.assertEqual(sorted(flor.just_friends),[12,18,24,30])

    def test_find_contacts_errors(self):
        self.FS['mloc_users.04'] = [dict(id=404)]
        self.FS['mloc_users.03'] = [dict(id=503)]
        with _patch_twitter():
            self.gob.run_job('find_contacts')
        for uid in (404,503):
            missing = User.get_id(uid)
            self.assertEqual(missing.error_status,uid)
            self.assertEqual(missing.neighbors, None)
            self.assertEqual(missing.rfriends, None)
            self.assertEqual(Edges.get_id(uid), None)
            self.assertEqual(Tweets.get_id(uid), None)

    def test_contact_split(self):
        User(_id=24).save()
        self._find_contacts_6()
        self.gob.run_job('contact_split')
        self.assertEqual(self.FS['contact_split.18'],[18])
        self.assertNotIn('contact_split.24', self.FS)

    def _fake_find_contacts(self, *ids):
        self.FS['find_contacts'] = [
                (User.mod_id(uid),[uid])
                for uid in ids
                ]

    def _lookup_contacts(self):
        self.FS['mdists'] = [dict(other=2)]
        with _patch_twitter():
            with _patch_gisgraphy():
                self.gob.run_job('contact_split')
                self.gob.run_job('lookup_contacts')

    def test_lookup_contacts(self):
        self._fake_find_contacts(2,3)
        self._lookup_contacts()
        beryl = User.get_id(2)
        self.assertEqual(beryl.screen_name,'user_2')
        self.assertEqual(beryl.geonames_place.feature_code,'PPLA2')
        self.assertEqual(beryl.geonames_place.mdist,3)

    def test_pick_nebrs(self):
        User(_id=6, just_mentioned=[7], just_friends=[1,2,3]).save()
        User(_id=1, gnp=dict(mdist=1000)).save()
        User(_id=2, gnp=dict(mdist=5)).save()
        User(_id=3, gnp=dict(mdist=10)).save()
        nebrs = sprawl.pick_nebrs(6)
        self.assertEqual(list(nebrs),[2,3])

    def test_fix_mloc_mdists(self):
        self.FS['mdists'] = [dict(other=2)]
        self.FS['mloc_uids.03'] = [3,103]
        User(_id=3, location="Texas").save()
        User(_id=103, location="Bryan, TX").save()
        with _patch_gisgraphy():
            self.gob.run_job('fix_mloc_mdists')
        u3 = User.get_id(3)
        u103 = User.get_id(103)
        self.assertEqual(u3.geonames_place.mdist,2000)
        self.assertEqual(u103.geonames_place.mdist,2)

