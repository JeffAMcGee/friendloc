import os
import os.path
import json

import mock

from base.models import User
from base.tests import SimpleGobTest
from base.tests.models_tests import MockTwitterResource


class TestSprawlToContacts(SimpleGobTest):
    @classmethod
    def setUpClass(cls):
        CWD = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(CWD,'..','..','tests','geo_tweets.json')
        with open(path) as file:
            cls.geotweets = [json.loads(l) for l in file]

    def setUp(self):
        super(TestSprawlToContacts, self).setUp()
        self.env.save("geotweets", self.geotweets)

    def test_mloc_users(self):
        self.gob.run_job('mloc_users')
        results = self.FS['mloc_users.39']
        self.assertEqual(results[0]['id'], 51839)
        self.assertEqual(results[0]['mloc'], [-95.3,29.2])
        self.assertEqual(len(results), 1)

    def _find_edges_6(self):
        self.FS['mloc_users.06'] = [dict(id=6)]
        with mock.patch("base.twitter.TwitterResource",MockTwitterResource):
            # FIXME: mock gisgraphy?
            self.gob.run_job('find_edges')

    def test_find_edges(self):
        self._find_edges_6()
        results = self.FS['find_edges.06']
        s_res = sorted(list(r[1])[0] for r in results)
        self.assertEqual(s_res, [0,1,2,3,12,18,24,30])

    def test_contact_split(self):
        User(_id=24).save()
        self._find_edges_6()
        self.gob.run_job('contact_split')
        self.assertEqual(self.FS['contact_split.18'],[18])
        self.assertNotIn('contact_split.24', self.FS)

    #def test_lookup_contacts(self):
    #    self.FS['find_edges'] = [('02',[2]), ('03',[3])]
    #    self.gob.run_job('contact_split')
    #    self.gob.run_job('lookup_contacts')
