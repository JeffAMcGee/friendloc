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
    block = args[-1]
    #make predictior objects from class names in args
    pred_names = args[:-1]
    predictors = [globals()[p]() for p in pred_names]

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
        for predictor,label in zip(predictors,pred_names):
            res = predictor.pred(user)
            dists[label].append(coord_in_miles(mloc, res))
    write_json([dists],"samp/res"+block)
    print skipped

class Median():
    def pred(self, user):
        return median_2d((r['lng'],r['lat']) for r in user['rels'])


class Geocoding():
    def pred(self, user):
        return user['gnp']


class Omniscient():
    def pred(self, user):
        return min(user['rels'], key=lambda rel: coord_in_miles(user['mloc'],rel))

class MLE():
    def __init__(self):
        params = list(read_json("params"))[0]
        for k,v in params.iteritems():
            setattr(self,k,v)

    def pred(self, user):
        self.rels = user['rels']
        #redefine folc as the folc bin
        buckets = settings.fol_count_buckets
        for rel in self.rels:
            folc = max(rel['folc'],1)
            rel['folc'] = min(buckets-1, int(math.log(folc,4)))
        best_i, best = max(enumerate(self.rels), key=self._user_prob)
        return best

    def _user_prob(self, i_rel):
        i, rel = i_rel
        return sum(self._edge_prob(rel,edge) for j,edge in enumerate(self.rels) if i!=j)

    def _edge_prob(self,center,edge):
        dist = coord_in_miles(center,edge)
        kind, folc, mdist = edge['kind'], edge['folc'], edge['mdist']
        if dist<mdist:
            local = self.inner[kind][folc]/(mdist**2)
        else:
            a,b = self.a[kind][folc],self.b[kind][folc]
            local = (math.e**b) * (dist**a)
        rand = self.rand[kind][folc]/(2750**2)
        return math.log(local+rand)
