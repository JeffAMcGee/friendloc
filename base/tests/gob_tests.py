import unittest
import functools
from base import gob
from base.gob import Gob, Job
from base.gob import THE_FS


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
    def __init__(self,gob):
        pass

    @gob.func(gives_keys=True)
    def take(self, value):
        if 3 in value['digits']:
            yield value['num']%3, value['num']

    @gob.func(all_items=True)
    def results(self, items):
        # items should be an iterator of (k,v) pairs - just store the data on
        # the class so we can run tests on it
        self.__class__.results = dict(items)


def load_jobs(gob):
    gob.add_job(counter,saver=Job.split)
    gob.add_job(expand,'counter')
    gob.add_job(SecondHalf.take,'expand',saver=Job.list_reduce)
    gob.add_job(SecondHalf.results,'take')


class TestGob(unittest.TestCase):

    def test_counter(self):
        gob = Gob()
        load_jobs(gob)
        #THE_FS['expand'] = [
        #    dict(num=2,digits=(1,)),
        #    dict(num=12,digits=(1,2)),
        #    ]

        gob.run_job('counter')
        gob.run_job('expand')
        gob.run_job('take')
        gob.run_job('results')
        print SecondHalf.results

