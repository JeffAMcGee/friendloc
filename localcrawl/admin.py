import json
import time
import logging
import sys
from datetime import datetime as dt

import pymongo
try:
    import beanstalkc
except:
    pass

from settings import settings
from localcrawl.crawl import CrawlProcess
import localcrawl.lookup as lookup
from localcrawl.procs import create_slaves
from base.models import *
from maroon import Model
from base.utils import grouper, in_local_box, read_json


def localcrawl():
    while True:
        for region in settings.regions:
            crawl_once(region)
        time.sleep(600)


def crawl_once(region):
    proc = CrawlProcess(
            region,
            label=region,
            slaves=settings.slaves,
            log_level=logging.INFO)
    proc.run()


def start_lookup():
    print "spawning minions!"
    create_slaves(lookup.LookupSlave)
    proc = lookup.LookupMaster()
    proc.run()


def start_lookup_slave():
    proc = lookup.LookupSlave('x')
    proc.run()


def stop_lookup():
    stalk = beanstalkc.Connection()
    stalk.use(settings.region+"_lookup_done")
    stalk.put('halt',0)


def kick_all():
    stalk = beanstalkc.Connection()
    tube = settings.region+"_lookup"
    stalk.use(tube)
    stalk.kick(stalk.stats_tube(tube)['current-jobs-buried'])


def import_json():
    for g in grouper(1000,sys.stdin):
        try:
            Model.database.bulk_save([json.loads(l) for l in g if l])
        except BulkSaveError as err:
            if any(d['error']!='conflict' for d in err.errors):
                raise
            else:
                logging.warn("conflicts for %r",[d['id'] for d in err.errors])


def make_indexes(host=settings.mongo_host):
    keys = [('User','ncd'),
            ('Tweet',[('uid',1),('ca',1)]),
            ('Tweet',[('ats',1),('ca',1)]) ]
    connection = pymongo.Connection(host=host)
    for db in settings.regions:
        for key in keys:
            connection[db][key[0]].create_index(key[1])


def import_mongo(cls_name,path=None):
    Cls = globals()[cls_name]
    for g in grouper(1000,read_json(path)):
        Cls.bulk_save(Cls(from_dict=d) for d in g if d)


def fix_crowd():
    #FIXME: this is for crowdy
    for crowd in Model.database.Crowd.find():
        for k in ['start','end']:
            if not crowd[k]:
                crowd[k] = dt(2011,6,1)
            elif not isinstance(crowd[k],dt):
                crowd[k] = dt.utcfromtimestamp(crowd[k])
        end = crowd['end']
        for user in crowd['users']:
            user['id'] = int(user['id'])
            for h in user['history']:
                for x in xrange(2):
                    if not h[x]:
                        h[x] = end
                    elif not isinstance(h[x], dt):
                        h[x] = dt.utcfromtimestamp(h[x])
        Model.database.Crowd.save(crowd)


def pick_locals(path=None):
    file =open(path) if path else sys.stdin 
    for line in file:
        tweet = json.loads(line)
        if not tweet.get('coordinates'):
            continue
        d = dict(zip(['lng','lat'],tweet['coordinates']['coordinates']))
        if in_local_box(d):
            print json.dumps(tweet['user'])
