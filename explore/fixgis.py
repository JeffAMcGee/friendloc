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

def print_gnp_gps():
    users = {}
    for i,t in enumerate(utils.read_json()):
        if i%10000 ==0:
            logging.info("read %d tweets",i)
        if 'id' not in t: continue # this is not a tweet
        uid = t['user']['id']
        if not t.get('coordinates'): continue
        if uid%10>6: continue
        if uid not in users:
            users[uid] = dict(
                _id = t['user']['id'],
                loc = t['user'].get('location',''),
                locs = [],
            )
        users[uid]['locs'].append(t['coordinates']['coordinates'])
    logging.info("sending %d users",len(users))
    p = Pool(8)
    with open('gnp_gps','w') as f:
        items = p.imap_unordered(
                _lookup_user,
                users.itervalues(),
                chunksize=100)
        for item in items:
            if item:
                print>>f, json.dumps(item)

def _lookup_user(user):
    spots = user['locs']
    if len(spots)<2 or not user['loc']:
        return None
    median = utils.median_2d(spots)
    dists = [utils.coord_in_miles(median,spot) for spot in spots]
    if numpy.median(dists)>50:
        return None #user moves too much
    gnp = gis.twitter_loc(user['loc'])
    if not gnp:
        return None
    return dict(
        _id = user['_id'],
        loc = user['loc'],
        mloc = median,
        gnp = gnp.to_d(),
        )


def gisgraphy_mdist():
    mdist = {}
    dists = defaultdict(list)
    gnps = {}
    for u in utils.read_json('gnp_gps'):
        if u['_id']%10>3: continue
        d = utils.coord_in_miles(u['gnp'],u['mloc'])
        id = u['gnp'].get('feature_id','COORD')
        dists[id].append(d)
        gnps[id] = u['gnp']
    codes = defaultdict(list)
    for k,gnp in gnps.iteritems():
        if len(dists[k])>2:
            #add an entry for each feature that has a meaningful median
            mdist[str(k)] = numpy.median(dists[k])
        else:
            codes[gnp.get('code')].append(dists[k][0])
    other = []

    for k,code in codes.iteritems():
        if len(code)>5:
            #add an entry for each feature code that has a meaningful median
            mdist[k] = numpy.median(codes[k])
        else:
            other.extend(code)
    #add a catch-all for everything else
    mdist['other'] = numpy.median(other)
    utils.write_json([mdist],'mdists')


class GisgraphyMdist():
    def __init__(self):
        self._mdist = list(utils.read_json('mdists'))[0]

    def mdist(self,gnp):
        try:
            gnp = gnp.to_d()
        except AttributeError:
            pass
        id = str(gnp.get('feature_id',"COORD"))
        if id in self._mdist:
            return self._mdist[id]
        if gnp['code'] in self._mdist:
            return self._mdist[gnp['code']]
        return self._mdist['other']



