import os
import os.path
import json

from base.tests import SimpleGobTest


class TestSprawlToContacts(SimpleGobTest):
    @classmethod
    def setUpClass(cls):
        CWD = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(CWD,'..','..','tests','geo_tweets.json')
        print path
        with open(path) as file:
            cls.geotweets = [json.loads(l) for l in file]

    def setUp(self):
        super(TestSprawlToContacts, self).setUp()
        self.env.save("geotweets", self.geotweets)

    def test_mloc_users(self):
        self.gob.run_job('mloc_users')
        print self.FS
        results = self.FS['mloc_users.39']
        self.assertEqual(results[0]['id'], 51839)
        self.assertEqual(results[0]['mloc'], [-95.3,29.2])
        self.assertEqual(len(results), 1)
