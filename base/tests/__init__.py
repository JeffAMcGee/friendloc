import unittest

from mock import patch
from maroon import Model, MockDB

from gb import create_jobs
from base.gob import Gob, SimpleEnv


class MockedMongoTest(unittest.TestCase):
    def setUp(self):
        super(MockedMongoTest, self).setUp()
        dbpatch = patch.object(Model,'database',MockDB())
        dbpatch.start()
        self.addCleanup(dbpatch.stop)


class SimpleGobTest(unittest.TestCase):
    def setUp(self):
        super(SimpleGobTest, self).setUp()
        self.env = SimpleEnv()
        self.gob = Gob(self.env)
        create_jobs(self.gob)
        SimpleEnv.THE_FS = {}

    @property
    def FS(self):
        return SimpleEnv.THE_FS

