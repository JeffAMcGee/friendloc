import unittest

from mock import patch
from base.models import TwitterModel
from maroon import MockDB

from gb import create_jobs
from base.gob import Gob, SimpleEnv


class MockedMongoTest(unittest.TestCase):
    def setUp(self):
        super(MockedMongoTest, self).setUp()
        dbpatch = patch.object(TwitterModel,'database',MockDB())
        dbpatch.start()
        self.addCleanup(dbpatch.stop)


class SimpleGobTest(MockedMongoTest):
    def setUp(self):
        super(SimpleGobTest, self).setUp()
        self.env = SimpleEnv()
        self.gob = Gob(self.env)
        create_jobs(self.gob)
        SimpleEnv.THE_FS = {}
        SimpleEnv.JOB_DB = {}

    @property
    def FS(self):
        return SimpleEnv.THE_FS

