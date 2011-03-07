#!/usr/bin/env python

import sys
sys.path.append("..")

import unittest
import logging
import time
from splitproc import SplitProcess

class AdderSplitProc(SplitProcess):
    def __init__(self, delay, **kwargs):
        SplitProcess.__init__(self, **kwargs)
        self.delay = delay

    def produce(self):
        for x in xrange(10):
            time.sleep(self.delay)
            yield x

    def map(self,item):
        return item+1

    def consume(self,items):
        return sum(items)



class TestSplitProc(unittest.TestCase):

    def test_single(self):
        asp = AdderSplitProc(0)
        result = asp.run_single()
        self.assertEqual(55, result)

    def test_multi(self):
        self._run_test(delay=0,slaves=2)
        self._run_test(delay=0,slaves=15)
        self._run_test(delay=.2,slaves=2)

    def _run_test(self,**kwargs):
        asp = AdderSplitProc(**kwargs)
        result = asp.run_procs()
        self.assertEqual(55, result)


if __name__ == '__main__':
    unittest.main()
