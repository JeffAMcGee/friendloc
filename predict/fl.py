import math
from itertools import chain
import operator
from collections import defaultdict

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


class Predictor():
    def predict(self,nebrs_d):
        "return the index of a neighbor or (eventually) a coordinate"
        raise NotImplementedError


class Omni(Predictor):
    def predict(self,nebrs_d):
        values = (v[-1] for v in nebrs_d['vects'])
        index, dist = min(enumerate(values), key=operator.itemgetter(1))
        return index


class Nearest(Predictor):
    def __init__(self,env):
        self.env = env
        self.clf = next(env.load('nebr_clf','pkl'))

    def predict(self,nebrs_d):
        X, y = _transformed(nebrs_d['vects'])
        dists = self.clf.predict(X)
        index, dist = min(enumerate(dists), key=operator.itemgetter(1))
        return index

class FacebookMLE(Predictor):

    def predict(self,nebrs_d):
        X, y = _transformed(nebrs_d['vects'])
        dists = self.clf.predict(X)
        index, dist = min(enumerate(dists), key=operator.itemgetter(1))
        return index

class Predictors(object):
    def __init__(self,env):
        self.env = env
        self.classifiers = dict(
            omni=Omni(),
            nearest=Nearest(env),
        )

    @gob.mapper(all_items=True)
    def predictions(self, nebrs_ds):
        results = defaultdict(list)

        for nebrs_d in nebrs_ds:
            if not nebrs_d['nebrs']:
                continue
            nebrs_d['vects'] = list(nebr_vect(nebrs_d))
            for key,classifier in self.classifiers.iteritems():
                index = classifier.predict(nebrs_d)
                dist = nebrs_d['vects'][index][-1]
                results[key].append(unlogify(dist,.01))
        return results.iteritems()
