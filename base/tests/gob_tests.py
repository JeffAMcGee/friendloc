import unittest
from base import gob
from base.gob import Gob, Job, Storage


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


def load_jobs(gob):
    gob.add_job(counter,saver=Job.split)
    gob.add_job(expand,'counter')
    gob.add_job(SecondHalf.take,'expand',saver=Job.list_reduce)
    gob.add_job(SecondHalf.results,'take')


class TestGob(unittest.TestCase):

    def test_whole_gob(self):
        gob = Gob()
        load_jobs(gob)
        #Storage.THE_FS['expand'] = [
        #    dict(num=42,digits=(4,2)),
        #    dict(num=43,digits=(4,3)),
        #    ]

        gob.run_job('counter')
        gob.run_job('expand')
        gob.run_job('take')
        gob.run_job('results')

        # This job finds numbers less than 100 that begin with a four and end
        # with a 2 or 7.
        self.assertEqual(set(SecondHalf.results[2]), {42,47})

