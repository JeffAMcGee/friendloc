import math
from itertools import chain
import operator

from sklearn import preprocessing, tree, cross_validation
import numpy as np

from base import gob
from base.utils import coord_in_miles


def logify(x,fudge=1):
    return math.log(x+fudge,2)

def unlogify(x,fudge=1):
    return 2**x - fudge

def _scaled_local(x):
    return x if x is not None else .3

@gob.mapper()
def nebr_vect(user):
    for nebr in user['nebrs']:
        flags = [nebr['kind'] >>i & 1 for i in range(3)]
        logged = [logify(nebr[k]) for k in ('mdist','folc','frdc')]
        others = [
            _scaled_local(nebr['lofrd']),
            _scaled_local(nebr['lofol']),
            int(bool(nebr['prot'])),
            logify(coord_in_miles(user['mloc'],nebr),fudge=.01),
        ]
        yield flags + logged + others


def _transformed(vects):
    # convert vects to a scaled numpy array
    vects_ = np.fromiter( chain.from_iterable(vects), np.float32 )
    vects_.shape = (len(vects_)//10),10
    X = np.array(vects_[:,:-1])
    y = np.array(vects_[:,-1])
    #scaler = preprocessing.Scaler().fit(X)

    # In final system, you will want to only fit the training data.
    #return scaler.transform(X), y
    return X,y


@gob.mapper(all_items=True)
def nebr_clf(vects):
    X, y = _transformed(vects)
    clf = tree.DecisionTreeRegressor(max_depth=8)
    clf.fit(X,y)
    yield clf


class NebrRanker(object):
    def __init__(self,env):
        self.env = env
        self.clf = next(env.load('nebr_clf','pkl'))

    @gob.mapper()
    def nearest_nebr(self, nebrs_d):
        vects = list(nebr_vect(nebrs_d))
        if not vects:
            return
        omni = min(v[-1] for v in vects)
        X, y = _transformed(vects)
        results = zip(y,self.clf.predict(X))
        best = min(results, key=operator.itemgetter(1))
        yield unlogify(omni,.01), unlogify(best[0],.01)
