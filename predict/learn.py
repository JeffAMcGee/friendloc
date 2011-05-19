import itertools
import sys
import random
import logging
from collections import defaultdict
from bisect import bisect

import numpy

from settings import settings
from base.models import *
from base.utils import *


def learn_user_model(key='50'):
    use_mongo('usgeo')
    logging.info('started %s',key)
    Trainer().train(key)
    logging.info('saved %s',key)

def build_user_model():
    Trainer().reduce()

class Trainer():
    def __init__(self):
        self.bins = 10**numpy.linspace(0,1,11)
        self.buckets = settings.fol_count_buckets
        self.inner = numpy.zeros((16,self.buckets),numpy.int)
        self.power = numpy.zeros((16,self.buckets,len(self.bins)-1),numpy.int)
        self.total = numpy.zeros((16,self.buckets),numpy.int)

    def train(self, key):
        users = User.find(User.mod_group == int(key))
        for user in users:
            self.train_user(user)
        result = dict(
            inner = self.inner.tolist(),
            power = self.power.tolist(),
            total = self.total.tolist(),
            )
        write_json([result], "data/model%s"%key)

    def train_user(self,me):
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
        for group in grouper(100, lookups, dontfill=True):
            #get the users - this will be SLOW
            amigos = User.find(
                User._id.is_in(lookups) & User.geonames_place.exists(),
                fields =['gnp','folc','prot'],
                )
            for amigo in amigos:
                kind = _group(amigo._id) + (8 if amigo.protected else 0)
                self.add_edge(me.median_loc, amigo, kind)

    def add_edge(self, mloc, user, kind):
        gnp = user.geonames_place.to_d()
        dist = coord_in_miles(mloc,gnp)/gnp['mdist']

        folc = max(user.followers_count,1)
        bits = min(self.buckets-1, int(math.log(folc,4)))
        self.total[kind,bits]+=1
        
        if dist<1:
            self.inner[kind,bits]+=1
        else:
            bin = bisect(self.bins,dist)-1
            if bin < len(self.bins)-1:
                self.power[kind,bits,bin]+=1

    def reduce(self):
        for d in read_json("model"):
            for k in ('inner','power','total'):
                getattr(self,k).__iadd__(d[k])
        inner = numpy.true_divide(self.inner, self.total)

