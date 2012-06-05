#!/usr/bin/env python

if __name__ != '__main__':
    print """
This is a tool for testing and administrative tasks.  It is designed to
be %run in ipython or from the command line.  If you import it from another
module, you're doing something wrong.
"""

import logging
import argparse
import pdb
import os.path
import numpy

from base.tests import gob_tests
from base import gob


path = os.path.join(os.path.dirname(__file__),'data')
my_gob = gob.Gob(gob.MultiProcEnv(path))
gob_tests.create_jobs(my_gob)


logging.basicConfig(level=logging.INFO)
numpy.set_printoptions(precision=3, linewidth=160)

parser = argparse.ArgumentParser(description='Run gob job.')
parser.add_argument('job',nargs='+')
args = parser.parse_args()

for cmd in args.job:
    try:
        my_gob.run_job(cmd)
    except:
        logging.exception("command failed")
        pdb.post_mortem()
