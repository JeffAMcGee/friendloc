import unittest
import time
from friendloc.base.splitproc import SplitProcess


class AdderSplitProc(SplitProcess):
    def __init__(self, delay, **kwargs):
        SplitProcess.__init__(self, **kwargs)
        self.delay = delay

    def produce(self):
        for x in xrange(10):
            time.sleep(self.delay)
            yield x

    def map(self,items):
        for item in items:
            yield item+1

    def consume(self,items):
        return sum(items)


class TestSplitProc(unittest.TestCase):

    def test_single(self):
        asp = AdderSplitProc(0)
        result = asp.run_single()
        self.assertEqual(55, result)

    @unittest.skip
    def test_multi(self):
        self._run_test(delay=0,slaves=2)
        self._run_test(delay=0,slaves=15)
        self._run_test(delay=.2,slaves=2)

    def _run_test(self,**kwargs):
        asp = AdderSplitProc(**kwargs)
        result = asp.run()
        self.assertEqual(55, result)
