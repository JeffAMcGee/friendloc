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

    users = read_json("data/eval"+block)
    dists = defaultdict(list)

    for user in users:
        for predictor in preds:
            res = globals()['pred_'+predictor](user['rels'])
            dists[predictor].append(coord_in_miles(user['mloc'], res))
    write_json([dists],"data/res"+block)


def pred_median(rels):
    if rels:
        return median_2d((r['lng'],r['lat']) for r in rels)
    else:
        return settings.default_loc

