import os
import os.path
import shutil
import unittest

import mock
import msgpack

from base import gob
from base.gob import (Gob, SimpleEnv, SimpleFileEnv, MultiProcEnv,
                      join_reduce, set_reduce)


def source():
    return xrange(100)


@gob.mapper()
def counter(x):
    yield x%10,x


@gob.mapper()
def expand(value):
    yield dict(
        num=value,
        digits=[int(x) for x in str(value)])


class SecondHalf(object):
    result_data = {}

    def __init__(self,env):
        pass

    @gob.mapper()
    def take(self, value):
        if value['digits'][0]==4:
            yield value['num']%5, value['num']

    @gob.mapper(all_items=True)
    def results(self, items):
        # items should be an iterator of (k,v) pairs - just store the data on
        # the class so we can run tests on it
        self.__class__.result_data = dict(items)


def create_jobs(gob):
    gob.add_source(source)
    gob.add_job(counter,'source',saver='split_save')
    gob.add_job(expand,'counter')
    gob.add_job(SecondHalf.take,'expand',reducer=join_reduce)
    gob.add_job(SecondHalf.results,'take',saver=None)


class BaseGobTests(object):
    # This does not extend unittest.TestCase because I want these tests to run
    # in the subclasses to test all the environments.

    def test_whole_gob(self):
        # integration test
        self.gob.run_job('counter')
        self.gob.run_job('expand')
        self.gob.run_job('take')
        self.gob.run_job('results')

        # These jobs find numbers less than 100 that begin with a four and end
        # with a 2 or 7.
        results = dict(self.env.load('take'))
        self.assertEqual(set(results[2]), {42,47})

    def test_rerun(self):
        self.env.save("counter.2", range(2,100,10))

        with mock.patch.object(self.env,'map_single') as map_sing:
            map_sing.side_effect = ValueError()
            with self.assertRaises(Exception):
                self.gob.run_job('expand')
            status = self.env.job_status('expand.2')
            self.assertEqual(status, 'started')

            map_sing.side_effect = None
            self.gob.run_job('expand')
            status = self.env.job_status('expand.2')
            self.assertEqual(status, 'done')

            # since the job is done, map_single should not get called
            map_sing.side_effect = ValueError()
            self.gob.run_job('expand')


class TestSimpleEnv(unittest.TestCase, BaseGobTests):
    def setUp(self):
        self.env = SimpleEnv()
        self.gob = Gob(self.env)
        create_jobs(self.gob)
        SimpleEnv.THE_FS = {}
        SimpleEnv.JOB_DB = {}

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
        self.assertEqual( taken, [(2, (42, 47)), (3, (43,))] )

    def test_no_output(self):
        SimpleEnv.THE_FS['take'] = [(2, [42, 47]), (3, [43])]
        self.gob.run_job('results')
        self.assertEqual(set(SecondHalf.result_data[2]), {42,47})


def clean_data_dir():
    path = os.path.join(os.path.dirname(__file__),'data')
    shutil.rmtree(path,ignore_errors=True)
    os.mkdir(path)
    return path


class TestSimpleFileEnv(unittest.TestCase, BaseGobTests):
    UNICODE_STR = u'Unicode is \U0001F4A9!'

    def setUp(self):
        path = clean_data_dir()
        self.packed = msgpack.packb((1,2,"gigem",self.UNICODE_STR))

        self.env = SimpleFileEnv(path)
        self.gob = Gob(self.env)
        create_jobs(self.gob)

    def test_load(self):
        with open(os.path.join(self.env.path,"stuff.mp"),'w') as f:
            f.write(self.packed)

        data = list(self.env.load("stuff"))
        self.assertEqual(data,[(1,2,"gigem",self.UNICODE_STR)])
        self.assertTrue(isinstance(data[0][-1],unicode))

    def test_save(self):
        self.env.save("more_stuff.2",[[1,2,"gigem",self.UNICODE_STR]])
        with open(os.path.join(self.env.path,"more_stuff","2.mp")) as f:
            data = f.read()
        self.assertEqual(data,self.packed)

    def test_set_reduce(self):
        res = set_reduce(1,(2,4,5,2),rereduce=False)
        self.assertTrue(isinstance(res,tuple))
        self.assertEqual(sorted(res),[2,4,5])
        # make sure msgpack doesn't raise an exception
        msgpack.packb(res)


class TestMultiProcEnv(unittest.TestCase, BaseGobTests):
    def setUp(self):
        path = clean_data_dir()

        self.env = MultiProcEnv(path)
        self.gob = Gob(self.env)
        create_jobs(self.gob)

    def test_map(self):
        self.env.save('counter.2',xrange(2,100,10))
        self.env.save('counter.3',xrange(3,100,10))
        self.gob.run_job('expand')
        first = self.env.load('expand.2').next()
        self.assertEqual( first, {'digits':(2,),'num':2} )
