import itertools
import sys
import random
import logging
from collections import defaultdict
from multiprocessing import Pool

from settings import settings
from base.models import *
from base.utils import *

def eval_block(*args):
    preds, block = args[:-1],args[-1]

    users = read_json("samp/eval"+block)
    dists = defaultdict(list)
    skipped=0

    for user in users:
        mloc = user['mloc']
        if not user['rels']:
            skipped+=1
            continue
        if 'gnp' not in user or user['gnp']['code']=="COORD":
            user['gnp'] = settings.default_loc
        for predictor in preds:
            res = globals()['pred_'+predictor](user)
            dists[predictor].append(coord_in_miles(mloc, res))
    write_json([dists],"samp/res"+block)
    print skipped


def pred_median(user):
    return median_2d((r['lng'],r['lat']) for r in user['rels'])


def pred_geocoding(user):
    return user['gnp']


def pred_omniscient(user):
    return min(user['rels'], key=lambda rel: coord_in_miles(user['mloc'],rel))
    
