#!/usr/bin/env python

import logging
import argparse
import os.path
import numpy

from base.tests import gob_tests
from base import gob


def parse_args():
    parser = argparse.ArgumentParser(description='Run gob jobs.')
    parser.add_argument('job',nargs='+')
    parser.add_argument('-s','--single',action="store_true",
                        help='run in a single process')
    return parser.parse_args()


def make_gob(args):
    path = os.path.join(os.path.dirname(__file__),'data')
    if args.single:
        env = gob.SimpleFileEnv(path)
    else:
        env = gob.MultiProcEnv(path)
    my_gob = gob.Gob(env)
    gob_tests.create_jobs(my_gob)
    return my_gob


def setup(args):
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
