#!/usr/bin/env python
import numpy
import random
import logging
import itertools
import json
from multiprocessing import Pool 
from operator import itemgetter
from collections import defaultdict

import maroon

from base.gisgraphy import GisgraphyResource
import base.utils as utils
from settings import settings

gis = GisgraphyResource()

def print_gnp_gps(eval=False):
    start,end = (4,6) if eval else (0,3)
    users = {}
    for i,t in enumerate(utils.read_json()):
        if i%10000 ==0:
            logging.info("read %d tweets",i)
        if 'id' not in t: continue # this is not a tweet
        uid = t['user']['id']
        if not t.get('coordinates'): continue
        if not (start<=uid%10<=end): continue
        if uid not in users:
            users[uid] = dict(
                _id = t['user']['id'],
                loc = t['user'].get('location',''),
                locs = [],
            )
        users[uid]['locs'].append(t['coordinates']['coordinates'])
    logging.info("sending %d users",len(users))
    gnp_users = lookup_gnp_multi(_calc_mloc(u) for u in users.itervalues())
    utils.write_json(gnp_users,"gnp_gps_%d%d"%(start,end))


def relookup_gnp_gps(eval=False):
    #this is going away
    users = [u for u in utils.read_json('gnp_gps') if start<=u['_id']%10<=end]
    gnp_users = lookup_gnp_multi(users)
    utils.write_json(gnp_users,"gnp_gps_%d%d"%(start,end))


def lookup_gnp_multi(users):
    p = Pool(8)
    items = p.map(
        _lookup_gnp,
        itertools.ifilter(None,users),
        chunksize=100)
    return itertools.ifilter(None,items)


def _lookup_gnp(user):
    gnp = gis.twitter_loc(user['loc'])
    if not gnp:
        return None
    user['gnp'] = gnp.to_d()
    return user


def _calc_mloc(user):
    spots = user['locs']
    if len(spots)<2 or not user['loc']:
        return None
    median = utils.median_2d(spots)
    dists = [utils.coord_in_miles(median,spot) for spot in spots]
    if numpy.median(dists)>50:
        return None #user moves too much
    return dict(
        _id = user['_id'],
        loc = user['loc'],
        mloc = median,
        )

def gisgraphy_mdist(item_cutoff=2,kind_cutoff=5):
    mdist = {}
    dists = defaultdict(list)
    gnps = {}
    for u in utils.read_json('gnp_gps_03'):
        d = utils.coord_in_miles(u['gnp'],u['mloc'])
        id = u['gnp'].get('fid','COORD')
        dists[id].append(d)
        gnps[id] = u['gnp']
    codes = defaultdict(list)
    for k,gnp in gnps.iteritems():
        if len(dists[k])>item_cutoff:
            #add an entry for each feature that has a meaningful median
            mdist[str(k)] = numpy.median(dists[k])
        else:
            codes[gnp.get('code')].append(dists[k][0])
    other = []

    for k,code in codes.iteritems():
        if len(code)>kind_cutoff:
            #add an entry for each feature code that has a meaningful median
            mdist[k] = numpy.median(codes[k])
        else:
            other.extend(code)
    #add a catch-all for everything else
    mdist['other'] = numpy.median(other)
    utils.write_json([mdist],'mdists')
