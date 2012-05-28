import unittest
from base import gob
from base.gob import Gob, Job, DictStorage


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
    results = {}

    def __init__(self,job,storage):
        pass

    @gob.func(gives_keys=True)
    def take(self, value):
        if value['digits'][0]==4:
            yield value['num']%5, value['num']

    @gob.func(all_items=True,must_output=False)
    def results(self, items):
        # items should be an iterator of (k,v) pairs - just store the data on
        # the class so we can run tests on it
        print items
        self.__class__.results = dict(items)


class TestGob(unittest.TestCase):
    def setUp(self):
        self.gob = Gob()
        self.gob.add_job(counter,saver=Job.split)
        self.gob.add_job(expand,'counter')
        self.gob.add_job(SecondHalf.take,'expand',saver=Job.list_reduce)
        self.gob.add_job(SecondHalf.results,'take')
        DictStorage.THE_FS = {}

    def test_split_saver(self):
        self.gob.run_job('counter')
        self.assertEqual(DictStorage.THE_FS['counter.2'],range(2,100,10))

    def test_split_load(self):
        DictStorage.THE_FS['counter.2'] = range(2,100,10)
        DictStorage.THE_FS['counter.3'] = range(3,100,10)
        self.gob.run_job('expand')

        exp2 = DictStorage.THE_FS['expand.2']
        self.assertEqual( exp2[0], {'digits':[2],'num':2} )
        self.assertEqual( len(exp2), 10 )
        exp3 = DictStorage.THE_FS['expand.3']
        self.assertEqual( exp3[9], {'digits':[9,3],'num':93} )

    @unittest.skip
    def test_list_reduce(self):
        DictStorage.THE_FS['expand.2'] = [
            dict(num=32,digits=(3,2)),
            dict(num=42,digits=(4,2)),
            ]
        DictStorage.THE_FS['expand.3'] = [
            dict(num=43,digits=(4,3)),
            dict(num=53,digits=(5,3)),
            ]
        self.gob.run_job('take')

    @unittest.skip
    def test_whole_gob(self):
        self.gob.run_job('counter')
        self.gob.run_job('expand')
        self.gob.run_job('take')
        self.gob.run_job('results')

        # These jobs find numbers less than 100 that begin with a four and end
        # with a 2 or 7.
        self.assertEqual(set(SecondHalf.results[2]), {42,47})
