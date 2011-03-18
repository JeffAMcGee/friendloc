#!/usr/bin/env python
# This is a tool for investigating the collected data.  It is designed to
# be %run in ipython.  If you import it from another module, you're doing
# something wrong.

import logging
from collections import defaultdict
from operator import itemgetter

from settings import settings
import base.twitter
from base.models import *
from maroon import ModelCache, Model
from scoredict import Scores, BUCKETS, log_score


def place_tweets(start, end):
    return Model.database.paged_view('tweet/plc',
            include_docs=True,
            startkey=None,
            endkey=None,
            startkey_docid=start,
            endkey_docid=end,
            )


def count_users(key):
    counts = defaultdict(int)
    for u in all_users():
        counts[u['doc'].get(key,None)]+=1
    for k in sorted(counts.keys()):
        print "%r\t%d"%(k,counts[k])


def count_locations(path='counts'):
    counts = defaultdict(int)
    for u in all_users():
        if u['doc'].get('prob',0)==1:
            loc = u['doc'].get('loc',"")
            norm = " ".join(re.split('[^0-9a-z]+', loc.lower())).strip()
            counts[norm]+=1
    f = open(path,'w')
    for k,v in sorted(counts.iteritems(),key=itemgetter(1)):
        print>>f, "%r\t%d"%(k,v)
    f.close()


def count_tweets_in_box(start='T',end='U'):
    counts = defaultdict(int)
    box = settings.local_box
    for row in place_tweets(start,end):
        if 'coord' in row['doc']:
            c = row['doc']['coord']['coordinates']
            if box['lng'][0]<c[0]<box['lng'][1] and box['lat'][0]<c[1]<box['lat'][1]:
                counts['inb']+=1
            else:
                counts['outb']+=1
        else:
            counts['noco']+=1
    print dict(counts)


def read_locs(path=None):
    for l in open(path or "locs_uid"):
        s = l.split()
        yield float(s[0]),float(s[1]),s[2]
    logging.info("read points")


def _read_gis_locs(path=None):
    for u in _read_json(path or "hou_tri_users"):
        yield u['lng'],u['lat']


def analyze():
    "Find out how the scoring algorithm did."
    scores = Scores()
    scores.read(settings.lookup_out)

    locs = (0,.5,1)
    weights =(.1,.3,.5,.7,.9)
    counts = dict(
        (score, dict(
            (loc, dict(
                (weight,0)
                for weight in weights))
            for loc in locs))
        for score in xrange(BUCKETS))

    for user in User.get_all():
        if user.utc_offset==7200: continue
        state, rfs, ats = scores.split(user._id)

        for weight in weights:
            score = log_score(rfs,ats,weight)
            counts[score][user.local_prob][weight]+=1

    print "non\t\t\t\t\tunk\t\t\t\t\tlocal"
    for score in xrange(BUCKETS):
        for loc in locs:
            for weight in weights:
                print "%d\t"%counts[score][loc][weight],
        print


def _parse_ats(ats_path):
    ats =defaultdict(lambda: defaultdict(int))
    ated =defaultdict(lambda: defaultdict(int))
    for line in open(ats_path):
        uid,at = [int(i) for i in line.strip().split('\t')]
        ats[uid][at]+=1
        ated[at][uid]+=1
    return ats,ated


def mainstream_edges(edges):
    return [ e for e in edges
        if 47<=e['lmfrd']+e['lmfol']<=300
        if 101<=e['lyfrd']+e['lyfol']<=954
        ]


def edge_dist(users_path="hou_tri_users"):
    users, uids = _tri_users_dict_set(users_path)
    logging.info("looking at %d users",len(users))
    for u in users:
        obj = Edges.get_id(int(u))
        if obj._id==None:
            continue


