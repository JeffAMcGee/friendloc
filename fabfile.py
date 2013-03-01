import os.path

from fabric.api import local, lcd

from maroon import Model
from friendloc.base import utils
from friendloc.base.tests import models_tests


CWD = os.path.abspath(os.path.dirname(__file__))


def fixture():
    Model.database = utils.mongo('fl_fixture')
    models_tests.save_fixtures()


def mongod():
    with lcd(CWD):
        # FIXME: check that it has not started?
        local("mkdir -p .mongo")
        local("mongod --fork --dbpath .mongo --logappend --logpath .mongo/log")

