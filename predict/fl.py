import math
from itertools import chain
import operator
import bisect
import itertools
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

class Last(Predictor):
    def predict(self,nebrs_d):
        return len(nebrs_d['strange_prob'])-1

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
    def __init__(self,env,loc_factor,tz_factor=0,force_loc=False):
        self.env = env
        self.cutoffs,self.curves = zip(*tuple(self.env.load('vect_fit')))
        self.loc_factor = loc_factor
        self.tz_factor = tz_factor
        self.force_loc = force_loc

    def predict(self,nebrs_d):
        if self.force_loc and nebrs_d['gnp'] and nebrs_d['gnp']['mdist']<25:
            return len(nebrs_d['vects'])
        mat = nebrs_d['dist_mat']
        probs = np.empty_like(mat)
        for index,pred in enumerate(nebrs_d['pred_dists']):
            curve = self.curves[bisect.bisect(self.cutoffs,pred)-1]
            probs[index,:] = explore.peek.contact_curve(mat[index,:],*curve)
        total_probs = (
            np.sum(np.log(probs),axis=0) +
            nebrs_d['strange_prob'] +
            self.tz_factor*nebrs_d['tz_prob'] +
            self.loc_factor*nebrs_d['location_prob']
            )
        return np.argmax(total_probs)


class FriendLocWeights(Predictor):
    def __init__(self,env,weights):
        self.env = env
        self.cutoffs,self.curves = zip(*tuple(self.env.load('vect_fit')))
        self.weights = weights

    def predict(self,nebrs_d):

        mat = nebrs_d['dist_mat']
        probs = np.empty_like(mat)
        for index,pred in enumerate(nebrs_d['pred_dists']):
            curve = self.curves[bisect.bisect(self.cutoffs,pred)-1]
            probs[index,:] = explore.peek.contact_curve(mat[index,:],*curve)
        prob_sum = np.sum(np.log(probs),axis=0)
        w = self.weights
        total_probs = (
            w[0]*prob_sum +
            w[1]*prob_sum/len(nebrs_d['nebrs']) +
            w[2]*nebrs_d['strange_prob'] +
            w[3]*nebrs_d['tz_prob'] +
            w[4]*nebrs_d['location_prob']
            )
        return np.argmax(total_probs)


def _calc_dists(nebrs_d):
    gnp = nebrs_d['gnp']
    lats = [r['lat'] for r in nebrs_d['nebrs']]
    lngs = [r['lng'] for r in nebrs_d['nebrs']]
    all_lats = lats+[gnp['lat']] if gnp else lats
    all_lngs = lngs+[gnp['lng']] if gnp else lngs
    lat1,lat2 = np.meshgrid(all_lats,lats)
    lng1,lng2 = np.meshgrid(all_lngs,lngs)
    return utils.np_haversine(lng1,lng2,lat1,lat2)


class Predictors(object):
    def __init__(self,env):
        self.env = env
        self.classifiers = dict(
            omni=Omni(),
            nearest=Nearest(),
            last=Last(),
            backstrom=FacebookMLE(),
            friendloc_plain=FriendLoc(env,0),
            friendloc_field=FriendLoc(env,1),
            friendloc_tz=FriendLoc(env,1,1),
            friendloc_cut=FriendLoc(env,0,force_loc=True),
        )
        '''
        steps = [0,.5,1,2,25]
        self.classifiers = {
            vect:FriendLocWeights(env,vect)
            for vect in itertools.product(steps,repeat=5)
        }
        '''
        self.nebr_clf = next(env.load('nebr_clf','pkl'))

    def _mdist_curves(self):
        data = list(self.env.load('mdist_curves','mp'))
        return dict(
            curve_fn = [np.poly1d(d['coeffs']) for d in data],
            cutoff = [d['cutoff'] for d in data],
            local = [d['local'] for d in data],
        )

    def _stranger_mat(self):
        mat = next(self.env.load('stranger_mat','npz'))
        # strangers is based on contacts, but prediction is based on nebrs, so
        # we scale down the values in stranger_mat
        return np.maximum(-50,mat)*.25

    def _add_location_prob(self,nebrs_d):
        gnp = nebrs_d['gnp']
        if not gnp:
            nebrs_d['location_prob'] = np.zeros_like(nebrs_d['strange_prob'])
            return

        # append 0 to include the distance to the gnp point
        dists = np.append(nebrs_d['dist_mat'][:,-1],[0])
        mdist = nebrs_d['gnp']['mdist']
        index = bisect.bisect(self.mdist_curves['cutoff'],mdist)-1
        curve = self.mdist_curves['curve_fn'][index]
        local = np.log(self.mdist_curves['local'][index])
        probs = [
            (local if dist<1 else curve(np.log(dist)))
            for dist in dists
        ]
        nebrs_d['location_prob'] = np.array(probs)

    def _stranger_prob(self,nebr):
        lng_t = explore.peek._tile(nebr['lng'])
        lat_t_ = explore.peek._tile(nebr['lat'])
        lat_t = max(-900,min(lat_t_,899))
        return self.stranger_mat[(lng_t+1800)%3600,lat_t+900]

    def _tz_prob(self,nebrs_d,nebr):
        if 'utco' not in nebrs_d:
            return 1.0/24
        delta_lng = int(nebr['lng']-nebrs_d['utco']/240)
        index = ((delta_lng+180)%360)/15
        return self.utc_offset[index]

    def prep(self,nebrs_d):
        # add fields to nebrs_d
        nebrs_d['vects'] = list(nebr_vect(nebrs_d))
        X, y = vects_as_mat(nebrs_d['vects'])
        nebrs_d['pred_dists'] = self.nebr_clf.predict(X)
        nebrs_d['dist_mat'] = _calc_dists(nebrs_d)
        nebrs_d['contact_prob'] = utils.contact_prob(nebrs_d['dist_mat'])

        s_probs = [self._stranger_prob(nebr) for nebr in nebrs_d['nebrs']]
        if nebrs_d['gnp']:
            s_probs.append(self._stranger_prob(nebrs_d['gnp']))
        nebrs_d['strange_prob'] = np.array(s_probs)

        t_probs = [self._tz_prob(nebrs_d, nebr) for nebr in nebrs_d['nebrs']]
        if nebrs_d['gnp']:
            t_probs.append(self._tz_prob(nebrs_d, nebrs_d['gnp']))
        nebrs_d['tz_prob'] = np.log(t_probs)

        self._add_location_prob(nebrs_d)

    @gob.mapper(all_items=True)
    def predictions(self, nebrs_ds):
        results = defaultdict(list)
        self.stranger_mat = self._stranger_mat()
        self.utc_offset = next(self.env.load('utc_offset','mp'))
        self.mdist_curves = self._mdist_curves()

        for nebrs_d in nebrs_ds:
            if not nebrs_d['nebrs']:
                continue
            self.prep(nebrs_d)
            for key,classifier in self.classifiers.iteritems():
                index = classifier.predict(nebrs_d)
                if index==len(nebrs_d['vects']):
                    # the last one is the gnp one
                    dist = utils.coord_in_miles(nebrs_d['gnp'],nebrs_d['mloc'])
                else:
                    dist = unlogify(nebrs_d['vects'][index][-1],.01)
                results[key].append(dist)
        return results.iteritems()
