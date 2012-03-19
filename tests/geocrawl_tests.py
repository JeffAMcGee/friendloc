#!/usr/bin/env python

import sys
sys.path.append("..")

import unittest

from explore.geocrawl import GeoLookup
from base.models import Edges, User, Tweets

class TestSplitProc(unittest.TestCase):

    def test_produce(self):
        gl = GeoLookup("geo_tweets.json","test")
        results = list(gl.produce())
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['id'], 51839)
        self.assertEqual(results[0]['mloc'], [-95.3,29.2])

    def test_map(self):
        test_db = "geocrawl_test"
        info_d = {
            "screen_name":"infobotter",
            "id":90333071,
            "mloc":[-96,30],
            }
        gl = GeoLookup("geo_tweets.json",test_db)
        # FIXME: This test talks to Twitter, and will fail if data on Twitter
        # changes!
        list(gl.map([info_d]))
        infobotter = User.get_id(90333071)
        self.assertEqual(len(infobotter.rfriends), 1)
        self.assertEqual(infobotter.rfriends[0],48479480)
        self.assertEqual(len(infobotter.neighbors),3)
        all_edges = list(Edges.get_all())
        self.assertEqual(len(all_edges), 4)
        ib_tweets = Tweets.get_id(90333071)
        self.assertEqual(len(ib_tweets.ats), 2)


if __name__ == '__main__':
    #logging.basicConfig(level=logging.INFO)
    unittest.main()
