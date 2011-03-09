#!/usr/bin/env python

import sys
sys.path.append("..")

import logging
import unittest

from geocrawl import GeoLookup
from localcrawl.models import Edges, User, Tweet
from maroon import MongoDB, Model


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
        list(gl.map([info_d]))
        Model.database = MongoDB(name=test_db)
        infobotter = User.get_id(90333071)
        self.assertEqual(len(infobotter.rfriends), 1)
        self.assertEqual(infobotter.rfriends[0],48479480)


if __name__ == '__main__':
    #logging.basicConfig(level=logging.INFO)
    unittest.main()
