
import beanstalkc
import os
import string
import logging
from multiprocessing import Process

from settings import settings
from maroon import CouchDB, MongoDB, TeeDB
from base.models import *


class LocalProc(object):
    def __init__(self, task, slave_id=""):
        self.stalk = beanstalkc.Connection(
                settings.beanstalk_host,
                settings.beanstalk_port,
                )
        label = settings.region+"_"+task
        self.stalk.watch(label if slave_id else label+"_done")
        self.stalk.use(label+"_done" if slave_id else label)

        log = label+"_"+slave_id if slave_id else label
        filepath = os.path.join(settings.log_dir, log)
        logging.basicConfig(filename=filepath+".log",level=logging.INFO)
        
        if settings.db == 'couch':
            Model.database = CouchDB(settings.couchdb_root+settings.region,True)
        else:
            Model.database = TeeDB(filepath+".json",MongoDB(name=settings.region))


def _run_slave(Proc,slave_id,*args):
    p = Proc(slave_id,*args)
    try:
        p.run()
    except:
        logging.exception("exception killed proc")
        print "exception killed proc"


def create_slaves(Proc, *args, **kwargs):
    for x in xrange(settings.slaves):
        slave_id = kwargs.get("prefix","")+string.letters[x]
        run_args = (Proc,slave_id)+args
        p = Process(target=_run_slave, args=run_args)
        p.start()
