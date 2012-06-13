import os.path

from fabric.api import local, lcd


CWD = os.path.abspath(os.path.dirname(__file__))


def fixture():
    with lcd(CWD):
        # this is stupid
        local("./do.py save_fixtures -m fl_fixture")


def mongod():
    with lcd(CWD):
        # FIXME: check that it has not started?
        local("mkdir -p .mongo")
        local("mongod --fork --dbpath .mongo --logappend --logpath .mongo/log")

