import unittest
from base.job import Gob

from maroon import *


def counter():
    for x in xrange(1000):
        yield x%10,x

def primes():
    return (2,3,5,7,11,13,17,19,23,29,31)

def expand(value):
    yield dict(
        num=value,
        digits=[int(x) for x in str(value)])

def take(value):
    if 3 in value['digits']:
        yield value['num']%3, value['num']


class Recipient(object):
    results = {}

    def run(cls, items):
        # items should be an iterator of (k,v) pairs - just store the data on
        # the class so we can run tests on it
        cls.results = dict(items)


def load_jobs(gob):
    prod = gob.producer(counter)
    #exp = gob.maper(expand,prod)
    #taken = gob.list_reducer(take,exp)
    #gob.consumer(Recipient,taken)


class TestGob(unittest.TestCase):

    def test_counter(self):
        gob = Gob()
        load_jobs(gob)
        data = list(gob.run_job('counter'))
        self.assertEqual(data[0], (0,0))
        self.assertEqual(data[1], (1,1))
        self.assertEqual(data[72], (2,72))

