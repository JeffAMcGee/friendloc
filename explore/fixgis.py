#!/usr/bin/env python
import numpy
import random
import logging
import itertools
import json
from multiprocessing import Pool 

import maroon

from base.gisgraphy import GisgraphyResource
import base.utils as utils
from settings import settings

gis = GisgraphyResource()

def print_gnp_gps():
    users = {}
    for i,t in enumerate(utils.read_json('tests/geo_tweets.json')):
        if i%10000 ==0:
            logging.info("read %d tweets"%i)
        if 'id' not in t: continue # this is not a tweet
        uid = t['user']['id']
        if not t.get('coordinates'): continue
        if uid not in users:
            users[uid] = dict(
                _id = t['user']['id'],
                locs = [],
                loc = t['user'].get('location',''),
            )
        users[uid]['locs'].append(t['coordinates']['coordinates'])

    p = Pool(8)
    with open('gnp_gps','w') as f:
        for item in p.imap_unordered(_lookup_user,users.itervalues()):
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
    user['mloc'] = median
    user['gnp'] = gnp.to_d()
    return user
