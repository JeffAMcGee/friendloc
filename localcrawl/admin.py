import json
import itertools
import time
import logging
import sys
import heapq
import getopt
import sys
import cjson
from datetime import datetime as dt
import pdb

from couchdbkit import ResourceNotFound, BulkSaveError
import restkit.errors
from couchdbkit.loaders import FileSystemDocsLoader
import beanstalkc

from settings import settings
from scoredict import Scores, BUCKETS, log_score, DONE
import lookup
import twitter
from models import *
from utils import grouper, couch, mongo, in_local_box

def design_sync(type):
    "sync the documents in _design"
    loader = FileSystemDocsLoader(type+'_design')
    loader.sync(Model.database, verbose=True)


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


def filter_usians(path=None):
    file =open(path) if path else sys.stdin 
    for line in file:
        tweet = cjson.decode(line)
        if not tweet.get('coordinates'):
            continue
        lng,lat = tweet['coordinates']['coordinates']
        if 24<lat<50 and -126<lng<-66:
            print line,

def import_old_json():
    for g in grouper(1000,sys.stdin):
        docs = [json.loads(l) for l in g if l]
        for d in docs:
            del d['doc_type']
            for k,v in d.iteritems():
                if k[-2:]=='id' or k in ('rtt','rtu'):
                    d[k]=v[1:]
            for field in ['ats','fols','frs']:
                if field in d and isinstance(d[field],list):
                    d[field] = [u[1:] for u in d[field]]
        Model.database.bulk_save(docs)


def export_json(start=None,end=None):
    for d in Model.database.paged_view('_all_docs',include_docs=True,startkey=start,endkey=end):
        if d['id'][0]!='_':
            del d['doc']['_rev']
            print json.dumps(d['doc'])
 

def export_mongo():
    for d in Model.database.paged_view('_all_docs',include_docs=True):
        if d['id'][0]!='_':
            d = d['doc']
            del d['_rev']
            for k,v in d.iteritems():
                if k[-2:]=='id' or k in ('rtt','rtu'):
                    d[k]=int(v)
            for field in ['ats','fols','frs']:
                if field in d and isinstance(d[field],list):
                    d[field] = [int(u) for u in d[field][:5000]]
            print json.dumps(d)


def pick_locals(path=None):
    file =open(path) if path else sys.stdin 
    for line in file:
        tweet = json.loads(line)
        if not tweet.get('coordinates'):
            continue
        d = dict(zip(['lng','lat'],tweet['coordinates']['coordinates']))
        if in_local_box(d):
            print json.dumps(tweet['user'])


def merge_db(*names,**kwargs):
    views = [
        connect(name).paged_view(
            '_all_docs',
            include_docs=True,
            startkey=kwargs.get('start'),
            endkey=kwargs.get('end'))
        for name in names
    ]
    last =None
    for row in merge_views(*views):
        if row['key']!=last:
            del row['doc']['_rev']
            print json.dumps(row['doc'])
            last = row['key']


def fake_lu_master():
    proc = lookup.LookupMaster()
    while not proc.halt:
        proc.read_scores()
        print "scores:%d"%len(proc.scores)
    print "halting"
    proc.read_scores()
    proc.scores.dump(settings.lookup_out)


def fake_lu_slave():
    proc = lookup.LookupSlave('y')
    Edges.database = CouchDB('http://127.0.0.1:5984/orig_houtx',True)
    view = Model.database.paged_view('dev/user_and_tweets',include_docs=True)
    for k,g in itertools.groupby(view, lambda r:r['key'][0]):
        user_d = g.next()
        if user_d['id'][0] != 'U':
            print "fail %r"%user_d
            continue
        user = User(user_d['doc'])
        print "scoring %s - %s"%(user._id, user.screen_name)
        tweets = [Tweet(r['doc']) for r in g]
        if user.local_prob != 1.0:
            continue
        try:
            rels = Edges.get_for_user_id(user._id)
        except ResourceNotFound:
            print "rels not found"
            rels = None
        proc.score_new_users(user, rels, tweets)
    print "done"


def update_loc():
    User.database = connect("houtx_user")
    for user in User.get_all():
        old = user.local_prob
        user.local_prob = lookup.guess_location(user,gisgraphy)
        if old != user.local_prob:
            user.save()


def merge_views(*views):
    # This is based on heapq.merge in python 2.6.  The big difference is
    # that it sorts by key.
    h = []
    for itnum, it in enumerate(map(iter, views)):
        try:
            row = it.next()
            h.append([row['key'], itnum, row, it.next])
        except StopIteration:
            pass
    heapq.heapify(h)

    while 1:
        try:
            while 1:
                k, itnum, v, next = s = h[0]   # raises IndexError when h is empty
                yield v
                s[2] = next()               # raises StopIteration when exhausted
                s[0] = s[2]['key']
                heapq.heapreplace(h, s)          # restore heap condition
        except StopIteration:
            heapq.heappop(h)                     # remove empty iterator
        except IndexError:
            return


def _users_from_scores():
    scores = Scores()
    scores.read(settings.lookup_out)
    for uid in scores:
        state, rfs, ats = scores.split(uid)
        if log_score(rfs,ats)>=11:
            yield uid


def _users_from_db():
    User.database = connect("houtx_user")
    return (int(row['id']) for row in User.database.paged_view("_all_docs"))
    

def fetch_edges():
    Edges.database = connect("houtx_edges")
    User.database = connect("away_user")
    old_edges = set(int(row['id']) for row in Edges.database.paged_view("_all_docs",endkey="_"))
    uids = set(_users_from_scores())-old_edges
    for g in grouper(100,uids):
        for user in twitter.user_lookup(g):
            if user is None or user.protected: continue
            try:
                edges = twitter.get_edges(user._id)
            except restkit.errors.Unauthorized:
                logging.warn("unauthorized!")
                continue
            except restkit.errors.ResourceNotFound:
                logging.warn("resource not found!?")
                continue
            edges.save()
            user.save()
            twitter.sleep_if_needed()


def stdin_lookup():
    from_db = connect("orig_houtx")
    for l in sys.stdin:
        user = from_db.get_id(User,l.strip())
        user_lookup(user)


def user_lookup(user):
    tweets = twitter.save_timeline(user._id,last_tid=settings.min_tweet_id)
    if not tweets: return
    user.last_tid = tweets[0]._id
    user.last_crawl_date = dt.utcnow()
    user.next_crawl_date = dt.utcnow()
    user.tweets_per_hour = settings.tweets_per_hour
    user.lookup_done = True
    user.attempt_save()
    logging.info("saved %d from %s to %s",len(tweets),tweets[-1]._id,tweets[0]._id)
    twitter.sleep_if_needed()


def update_mongo():
    for user in User.get_all():
        try:                                                                             
            user.save()
        except:                                                                          
            pdb.post_mortem()    

def fill_50():
    Tweet.database = couch('hou_new_tweet')
    old_db = couch('houtx_tweet')
    res = twitter.TwitterResource()
    for line in open('logs/missing_uids'):
        uid = line.strip()
        view = old_db.paged_view('tweet/uid',key=uid)
        last = max(int(row['id']) for row in view)
        tweets = res.save_timeline(uid, last_tid=last)
        logging.info("saved %d for %s",len(tweets),uid)
        twitter.sleep_if_needed()

