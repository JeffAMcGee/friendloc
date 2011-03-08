#!/usr/bin/env python

import sys
sys.path.append("..")

import unittest
from geocrawl import GeoLookup


class TestSplitProc(unittest.TestCase):

    def test_produce(self):
        gl = GeoLookup("geo_tweets.json","test")
        results = list(gl.produce())
        print results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['id'], 51839)
        self.assertEqual(results[0]['mloc'], [-95.3,29.2])

    def test_reduce(self):
        infobotter = {
            "screen_name":"infobotter",
            "id":90333071,
            "mloc":[-96,30],
            }
        gl = GeoLookup("geo_tweets.json","test")
        list(gl.map([infobotter]))



if __name__ == '__main__':
    unittest.main()
