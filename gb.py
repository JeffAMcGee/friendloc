#!/usr/bin/env python

import logging
import argparse
import os.path
import numpy

from maroon import Model

from settings import settings
from explore import peek
from predict import prep, fl
from base import gob
from base import utils


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Run gob jobs.')
    parser.add_argument('job',nargs='+')
    parser.add_argument('-s','--single',action="store_true",
                        help='run in a single process')
    parser.add_argument('-m','--mongo')
    parser.add_argument('-i','--input')
    return parser.parse_args(argv)


def create_jobs(my_gob):
    my_gob.add_job(peek.geo_ats)
    my_gob.add_job(prep.mloc_users,saver='split_save')
    my_gob.add_job(prep.edge_d,'mloc_users')
    my_gob.add_job(fl.edge_vect,'edge_d')
    my_gob.add_job(fl.fl_learn,'edge_vect')


def make_gob(args):
    path = os.path.join(os.path.dirname(__file__),'data')
    if not args or args.single or args.input:
        env = gob.SimpleFileEnv(path)
    else:
        env = gob.MultiProcEnv(path)
    my_gob = gob.Gob(env)
    create_jobs(my_gob)
    return my_gob


def setup(args):
    Model.database = utils.mongo(args.mongo or settings.region)
    logging.basicConfig(level=logging.INFO)
    numpy.set_printoptions(precision=3, linewidth=160)


def inspect(job_name, source_set):
    """
    run a map reduce job on a single input and return its results.

    This is useful for poking around in ipython.
    """
    my_gob = make_gob(None)
    job = my_gob.jobs[job_name]
    funcs = job.runnable_funcs(my_gob.env)
    return my_gob.env.map_reduce_save(job, source_set, funcs)


def main(*argv):
    args = parse_args(argv or None)
    my_gob = make_gob(args)
    setup(args)
    if args.input:
        # just run the function for one input
        job = my_gob.jobs[args.job[0]]
        input_paths = [args.input.split(',')]
        my_gob.env.run(job,input_paths=input_paths)
    else:
        for cmd in args.job:
            my_gob.run_job(cmd)


if __name__ == '__main__':
    main()