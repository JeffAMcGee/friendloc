#!/usr/bin/env python

import logging
import argparse
import os.path
import numpy

from maroon import Model

from settings import settings
from explore import peek
from predict import prep
from base import gob
from base import utils


def parse_args():
    parser = argparse.ArgumentParser(description='Run gob jobs.')
    parser.add_argument('job',nargs='+')
    parser.add_argument('-s','--single',action="store_true",
                        help='run in a single process')
    parser.add_argument('-m','--mongo')
    return parser.parse_args()


def create_jobs(my_gob):
    my_gob.add_job(peek.geo_ats)
    my_gob.add_job(prep.mloc_users,saver='split_save')
    my_gob.add_job(prep.edges_d,'mloc_users')


def make_gob(args):
    path = os.path.join(os.path.dirname(__file__),'data')
    if args.single:
        env = gob.SimpleFileEnv(path)
    else:
        env = gob.MultiProcEnv(path)
    my_gob = gob.Gob(env)
    create_jobs(my_gob)
    return my_gob


def setup(args):
    # FIXME: set database for MultiProc?
    Model.database = utils.mongo(args.mongo or settings.region)
    logging.basicConfig(level=logging.INFO)
    numpy.set_printoptions(precision=3, linewidth=160)


def main():
    args = parse_args()
    my_gob = make_gob(args)
    setup(args)
    for cmd in args.job:
        my_gob.run_job(cmd)

if __name__ == '__main__':
    main()
