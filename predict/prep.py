import itertools
import sys
import random
import logging
from collections import defaultdict
from multiprocessing import Pool

from settings import settings
from base.models import *
from base.utils import use_mongo, write_json

def save_mod_group():
    for u in User.find(User.median_loc.exists()):
        u.mod_group = u._id%100
        u.save()

def prep_eval_users(key='50'):
    use_mongo('usgeo')
    logging.info('started %s',key)
    users = User.find(User.mod_group == int(key))
    write_json(itertools.imap(_edges_d, users), "data/eval"+key)
    logging.info('saved %s',key)


def _edges_d(me):
    #./do.py save_user_json edges_json 
    tweets = Tweets.get_id(me._id,fields=['ats'])
    edges = Edges.get_id(me._id)
    ats = set(tweets.ats or [])
    frds = set(edges.friends or [])
    fols = set(edges.followers or [])
    
    group_order = (ats,frds,fols)
    def _group(uid):
        #return 7 for ated rfrd, 4 for ignored jfol
        return sum(2**i for i,s in enumerate(group_order) if uid in s)

    #pick the 100 best users
    lookups = edges.lookups if edges.lookups else list(ats|frds|fols)
    random.shuffle(lookups)
    lookups.sort(key=_group, reverse=True)
    lookups = lookups[:100]

    #get the users - this will be SLOW
    amigos = User.find(
            User._id.is_in(lookups) & User.geonames_place.exists(),
            fields =['gnp','folc'],
            )

    rels = [_rel_d(amigo,_group(amigo._id)) for amigo in amigos]
    return dict(
            _id = me._id,
            mloc = me.median_loc,
            rels = rels,
            lu_len = len(lookups)
            )


def _rel_d(user, kind):
    gnp = user.geonames_place.to_d()
    return dict(
        folc=user.followers_count,
        lat=gnp['lat'],
        lng=gnp['lng'],
        mdist=gnp['mdist'],
        kind=kind,
        _id=user._id,
        )
