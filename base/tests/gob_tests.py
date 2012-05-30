import os.path
import unittest

from base import gob
from base.gob import Gob, SimpleEnv, SimpleFileEnv


@gob.func(gives_keys=True)
def counter():
    for x in xrange(100):
        yield x%10,x


@gob.func()
def expand(value):
    yield dict(
        num=value,
        digits=[int(x) for x in str(value)])


class SecondHalf(object):
    result_data = {}

    def __init__(self,job,storage):
        pass

    @gob.func(gives_keys=True)
    def take(self, value):
        if value['digits'][0]==4:
            yield value['num']%5, value['num']

    @gob.func(all_items=True)
    def results(self, items):
        # items should be an iterator of (k,v) pairs - just store the data on
        # the class so we can run tests on it
        self.__class__.result_data = dict(items)


def create_jobs(gob):
    gob.add_job(counter,saver='split')
    gob.add_job(expand,'counter')
    gob.add_job(SecondHalf.take,'expand',saver='list_reduce')
    gob.add_job(SecondHalf.results,'take',saver=None)


class TestSimpleEnv(unittest.TestCase):
    def setUp(self):
        self.gob = Gob(SimpleEnv())
        create_jobs(self.gob)
        SimpleEnv.THE_FS = {}

    def test_split_saver(self):
        self.gob.run_job('counter')
        self.assertEqual(SimpleEnv.THE_FS['counter.2'],range(2,100,10))

    def test_split_load(self):
        SimpleEnv.THE_FS['counter.2'] = range(2,100,10)
        SimpleEnv.THE_FS['counter.3'] = range(3,100,10)
        self.gob.run_job('expand')

        exp2 = SimpleEnv.THE_FS['expand.2']
        self.assertEqual( exp2[0], {'digits':[2],'num':2} )
        self.assertEqual( len(exp2), 10 )
        exp3 = SimpleEnv.THE_FS['expand.3']
        self.assertEqual( exp3[9], {'digits':[9,3],'num':93} )

    def test_list_reduce(self):
        SimpleEnv.THE_FS['expand.2'] = [
            dict(num=32,digits=(3,2)),
            dict(num=42,digits=(4,2)),
            dict(num=47,digits=(4,7)),
            ]
        SimpleEnv.THE_FS['expand.3'] = [
            dict(num=43,digits=(4,3)),
            dict(num=53,digits=(5,3)),
            ]
        self.gob.run_job('take')
        taken = SimpleEnv.THE_FS['take']
        self.assertEqual( taken, [(2, [42, 47]), (3, [43])] )

    def test_no_output(self):
        SimpleEnv.THE_FS['take'] = [(2, [42, 47]), (3, [43])]
        self.gob.run_job('results')
        self.assertEqual(set(SecondHalf.result_data[2]), {42,47})

    def test_whole_gob(self):
        # integration test
        self.gob.run_job('counter')
        self.gob.run_job('expand')
        self.gob.run_job('take')
        self.gob.run_job('results')

        # These jobs find numbers less than 100 that begin with a four and end
        # with a 2 or 7.
        self.assertEqual(set(SecondHalf.result_data[2]), {42,47})


class TestSimpleFileEnv(unittest.TestCase):
    def setUp(self):
        path = os.path.join(os.path.dirname(__file__),'data')
        self.env = SimpleFileEnv(path)
        self.gob = Gob(self.env)
        create_jobs(self.gob)

    def test_whole_gob(self):
        # integration test
        self.gob.run_job('counter')
        self.gob.run_job('expand')
        self.gob.run_job('take')
        self.gob.run_job('results')

        # These jobs find numbers less than 100 that begin with a four and end
        # with a 2 or 7.
        self.assertEqual(set(SecondHalf.result_data[2]), {42,47})
