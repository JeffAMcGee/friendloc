import math
from itertools import chain
import operator
import bisect
from collections import defaultdict

from sklearn import preprocessing, tree, cross_validation
import numpy as np

from base import gob, utils
from base.utils import coord_in_miles
import explore.peek


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


def vects_as_mat(vects):
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
    X, y = vects_as_mat(vects)
    clf = tree.DecisionTreeRegressor(max_depth=8)
    clf.fit(X,y)
    yield clf


class Predictor(object):
    def predict(self,nebrs_d):
        "return the index of a neighbor or (eventually) a coordinate"
        raise NotImplementedError


class Omni(Predictor):
    def predict(self,nebrs_d):
        values = [v[-1] for v in nebrs_d['vects']]
        return values.index(min(values))

class Nearest(Predictor):
    def predict(self,nebrs_d):
        return np.argmin(nebrs_d['pred_dists'])


class FacebookMLE(Predictor):
    def predict(self,nebrs_d):
        probs = np.sum(np.log(nebrs_d['contact_prob']),axis=0)
        return np.argmax(probs+nebrs_d['strange_prob'])


class NearestMLE(Predictor):
    def __init__(self,count):
        self.count = count

    def predict(self,nebrs_d):
        dists = sorted(
                    enumerate(nebrs_d['pred_dists']),
                    key=operator.itemgetter(1)
                    )
        best = [d[0] for d in dists[:self.count]]
        probs_ =    (
                    nebrs_d['contact_prob'][x][y]
                    for x in xrange(len(dists))
                    for y in xrange(len(dists))
                    if x in best and y in best
                    )
        probs = np.fromiter(probs_,float)
        probs.shape = len(best),len(best)

        sums = np.sum(np.log10(probs),axis=0)
        index, prob = max(enumerate(sums), key=operator.itemgetter(1))
        return best[index]


class FriendLoc(Predictor):
    def __init__(self,env):
        self.env = env

    def predict(self,nebrs_d):
        cutoffs,curves = zip(*tuple(self.env.load('vect_fit')))
        mat = nebrs_d['dist_mat']
        probs = np.empty_like(mat)
        for index,pred in enumerate(nebrs_d['pred_dists']):
            curve = curves[bisect.bisect(cutoffs,pred)-1]
            probs[index,:] = explore.peek.contact_curve(mat[index,:],*curve)
        total_probs = np.sum(np.log(probs),axis=0)
        return np.argmax(total_probs+nebrs_d['strange_prob'])


def _calc_dists(nebrs):
    lats = [r['lat'] for r in nebrs]
    lngs = [r['lng'] for r in nebrs]
    lat1,lat2 = np.meshgrid(lats,lats)
    lng1,lng2 = np.meshgrid(lngs,lngs)
    return utils.np_haversine(lng1,lng2,lat1,lat2)

class Predictors(object):
    def __init__(self,env):
        self.env = env
        self.classifiers = dict(
            omni=Omni(),
            nearest=Nearest(),
            nearest_5=NearestMLE(5),
            backstrom=FacebookMLE(),
            friendloc=FriendLoc(env),
        )
        self.nebr_clf = next(env.load('nebr_clf','pkl'))

    def _stranger_mat(self):
        mat = next(self.env.load('stranger_mat','npz'))
        # make nans go away
        for lat in range(1800):
            if np.isnan(mat[:,lat][0]):
                mat[:,lat] = mat[:,lat-1]
        # strangers is based on contacts, but prediction is based on nebrs, so
        # we scale down the values in stranger_mat
        return np.maximum(-50,mat)*.25

    def prep(self,nebrs_d):
        # add fields to nebrs_d
        nebrs_d['vects'] = list(nebr_vect(nebrs_d))
        X, y = vects_as_mat(nebrs_d['vects'])
        nebrs_d['pred_dists'] = self.nebr_clf.predict(X)
        nebrs_d['dist_mat'] = _calc_dists(nebrs_d['nebrs'])
        nebrs_d['contact_prob'] = utils.contact_prob(nebrs_d['dist_mat'])

        probs = []
        for nebr in nebrs_d['nebrs']:
            lng_t = explore.peek._tile(nebr['lng'])
            lat_t_ = explore.peek._tile(nebr['lat'])
            lat_t = max(-900,min(lat_t_,899))
            probs.append(self.stranger_mat[(lng_t+1800)%3600,lat_t+900])
        nebrs_d['strange_prob'] = np.array(probs)

    @gob.mapper(all_items=True)
    def predictions(self, nebrs_ds):
        results = defaultdict(list)
        self.stranger_mat = self._stranger_mat()

        for nebrs_d in nebrs_ds:
            if not nebrs_d['nebrs']:
                continue
            self.prep(nebrs_d)
            for key,classifier in self.classifiers.iteritems():
                index = classifier.predict(nebrs_d)
                dist = nebrs_d['vects'][index][-1]
                results[key].append(unlogify(dist,.01))
        return results.iteritems()
