import math
from itertools import chain
import operator
import bisect
from collections import defaultdict, Counter

from sklearn import tree
import numpy as np

from base import gob, utils
from base.utils import coord_in_miles
from predict import prep
import explore.peek


def logify(x,fudge=1):
    return math.log(x+fudge,2)


def unlogify(x,fudge=1):
    return 2**x - fudge


def _load_geo_ated(geo_ated):
    return {to:set(froms) for to, froms in geo_ated}


@gob.mapper(slurp={'geo_ated':_load_geo_ated,'dirt_cheap_locals':dict})
def nebr_vect(user,geo_ated,dirt_cheap_locals):
    mentioned = geo_ated.get(user['_id'],())
    for nebr in user['nebrs']:
        # I really don't like the way I did these flags.
        ated,fols,frds = [nebr['kind'] >>i & 1 for i in range(3)]
        at_back = int(nebr['_id'] in mentioned)
        flags = [ated, at_back, ated and at_back, fols, frds, fols and frds]
        logged = [logify(nebr[k]) for k in ('mdist','folc','frdc')]
        if 'mloc' in user:
            mloc_dist = logify(coord_in_miles(user['mloc'],nebr),fudge=.01)
        else:
            mloc_dist = float('nan')
        lorat = dirt_cheap_locals.get(nebr['_id'],.25)
        others = [ lorat, int(bool(nebr['prot'])), mloc_dist, ]
        yield flags + logged + others


def vects_as_mat(vects):
    # convert vects to a scaled numpy array
    vects_ = np.fromiter( chain.from_iterable(vects), np.float32 )
    vects_.shape = (len(vects_)//12),12
    X = np.array(vects_[:,:-1])
    y = np.array(vects_[:,-1])
    #scaler = preprocessing.Scaler().fit(X)

    # In final system, you will want to only fit the training data.
    #return scaler.transform(X), y
    return X,y


@gob.mapper(all_items=True)
def nebr_clf(vects):
    X, y = vects_as_mat(vects)
    # FIXME: I made up the number for min_samples_leaf
    clf = tree.DecisionTreeRegressor(min_samples_leaf=500)
    clf.fit(X,y)
    yield clf


class Predictor(object):
    def predict(self,nebrs_d,vect_fit):
        "return the index of a neighbor or (eventually) a coordinate"
        raise NotImplementedError


class Omni(Predictor):
    def predict(self,nebrs_d,vect_fit):
        values = [v[-1] for v in nebrs_d['vects']]
        return values.index(min(values))

class Nearest(Predictor):
    def predict(self,nebrs_d,vect_fit):
        return np.argmin(nebrs_d['pred_dists'])

class Last(Predictor):
    def predict(self,nebrs_d,vect_fit):
        return len(nebrs_d['strange_prob'])-1

class FacebookMLE(Predictor):
    def predict(self,nebrs_d,vect_fit):
        probs = np.sum(np.log(nebrs_d['contact_prob']),axis=0)
        return np.argmax(probs+nebrs_d['strange_prob'])

class Median(Predictor):
    def predict(self,nebrs_d,vect_fit):
        lats = [r['lat'] for r in nebrs_d['nebrs']]
        lngs = [r['lng'] for r in nebrs_d['nebrs']]
        mlat = np.median(lats)
        mlng = np.median(lngs)
        dists = utils.np_haversine(mlng,lngs,mlat,lats)
        return np.argmin(dists)

class Mode(Predictor):
    def predict(self,nebrs_d,vect_fit):
        spots = [(r['lng'],r['lat']) for r in nebrs_d['nebrs']]
        best,count = Counter(spots).most_common(1)[0]
        return spots.index(best)


class NearestMLE(Predictor):
    def __init__(self,count):
        self.count = count

    def predict(self,nebrs_d,vect_fit):
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
    def __init__(self,env,loc_factor=0,tz_factor=0,strange_factor=0,force_loc=False):
        self.env = env
        self.loc_factor = loc_factor
        self.strange_factor = strange_factor
        self.tz_factor = tz_factor
        self.force_loc = force_loc

    def predict(self,nebrs_d,vect_fit):
        self.cutoffs,self.curves = zip(*vect_fit)
        if self.force_loc and nebrs_d['gnp'] and nebrs_d['gnp']['mdist']<25:
            return len(nebrs_d['vects'])
        mat = nebrs_d['dist_mat']
        probs = np.empty_like(mat)
        for index,pred in enumerate(nebrs_d['pred_dists']):
            curve = self.curves[bisect.bisect(self.cutoffs,pred)-1]
            probs[index,:] = explore.peek.contact_curve(mat[index,:],*curve)
        total_probs = (
            np.sum(np.log(probs),axis=0) +
            self.strange_factor*nebrs_d['strange_prob'] +
            self.tz_factor*nebrs_d['tz_prob'] +
            self.loc_factor*nebrs_d['location_prob']
            )
        return np.argmax(total_probs)


class FriendLocWeights(Predictor):
    def __init__(self,env,weights,vect_fit):
        self.env = env
        self.weights = weights

    def predict(self,nebrs_d,vect_fit):
        self.cutoffs,self.curves = zip(*vect_fit)
        mat = nebrs_d['dist_mat']
        probs = np.empty_like(mat)
        for index,pred in enumerate(nebrs_d['pred_dists']):
            curve = self.curves[bisect.bisect(self.cutoffs,pred)-1]
            probs[index,:] = explore.peek.contact_curve(mat[index,:],*curve)
        prob_sum = np.sum(np.log(probs),axis=0)
        w = self.weights
        total_probs = (
            w[0]*prob_sum +
            w[1]*nebrs_d['location_prob'] +
            w[2]*nebrs_d['strange_prob']/2 +
            w[3]*nebrs_d['tz_prob']/2
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


@gob.mapper(all_items=True,slurp={'geo_ated':_load_geo_ated,'dirt_cheap_locals':dict})
def predictions(nebrs_ds, env, in_paths, geo_ated, dirt_cheap_locals):
    # This function is a stepping-stone to removing a useless class.
    p = Predictors(env)
    return p.predictions(nebrs_ds,in_paths,geo_ated,dirt_cheap_locals)



class Predictors(object):
    def __init__(self,env):
        self.env = env
        self.classifiers = dict(
            omni=Omni(),
            nearest=Nearest(),
            median=Median(),
            mode=Mode(),
            last=Last(),
            backstrom=FacebookMLE(),
            friendloc_plain=FriendLoc(env,0),
            friendloc_loc=FriendLoc(env,loc_factor=1),
            friendloc_tz=FriendLoc(env,tz_factor=1),
            friendloc_strange=FriendLoc(env,strange_factor=1),
            friendloc_cut0=FriendLoc(env,force_loc=True),
            friendloc_cut=FriendLoc(env,tz_factor=1,strange_factor=1,force_loc=True),
            friendloc_full=FriendLoc(env,1,1,1),
        )
        '''
        steps = [0,.333,1,3]
        self.classifiers = {
            vect:FriendLocWeights(env,vect)
            for vect in itertools.product(steps[1:],steps[1:],steps,steps)
            if any(vect)
        }
        '''

    def _mdist_curves(self,clump):
        data = list(self.env.load('mdist_curves.'+clump,'mp'))
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

    def prep_nebrs(self,nebrs_d):
        # add fields to nebrs_d
        vects = nebr_vect(nebrs_d,self.geo_ated,self.cheap_locals)
        nebrs_d['vects']=list(vects)
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

    def load_env(self,env,clump):
        # FIXME: clean this up
        self.utc_offset = next(self.env.load('utc_offset.'+clump,'mp'))
        self.mdist_curves = self._mdist_curves(clump)
        self.nebr_clf = next(self.env.load('nebr_clf.'+clump,'pkl'))
        self.stranger_mat = self._stranger_mat()
        self.vect_fit = tuple(self.env.load('vect_fit.'+clump))


    def predictions(self, nebrs_ds, in_paths, geo_ated, dirt_cheap_locals):
        results = defaultdict(list)
        clump = in_paths[0][-1]
        self.load_env(self.env,clump)

        self.geo_ated = geo_ated
        self.cheap_locals = dirt_cheap_locals

        for nebrs_d in nebrs_ds:
            if not nebrs_d['nebrs']:
                continue
            self.prep_nebrs(nebrs_d)
            for key,classifier in self.classifiers.iteritems():
                index = classifier.predict(nebrs_d,self.vect_fit)
                if index==len(nebrs_d['vects']):
                    # the last one is the gnp one
                    dist = utils.coord_in_miles(nebrs_d['gnp'],nebrs_d['mloc'])
                else:
                    dist = unlogify(nebrs_d['vects'][index][-1],.01)
                results[key].append(dist)
        return results.iteritems()

    def predict(self, user, nebrs, ats, ated):
        # this is an ugly way to deal with geo_ated and cheap_locals
        self.geo_ated = {user._id:ated}
        self.cheap_locals = {}
        for nebr in nebrs:
            nan = math.isnan(nebr.local_ratio)
            self.cheap_locals[nebr._id] = None if nan else nebr.local_ratio

        nebrs_d = prep.make_nebrs_d(user,nebrs,ats)
        self.prep_nebrs(nebrs_d)

        index = self.classifiers['friendloc_full'].predict(nebrs_d,self.vect_fit)
        if index==len(nebrs_d['nebrs']):
            return user.geonames_place.to_tup()
        else:
            dists = nebrs_d['dist_mat'][index]
            if np.median(dists)>1000:
                return None
            nebr = nebrs_d['nebrs'][index]
            return nebr['lng'],nebr['lat']


def _aed(ratio,vals):
    return np.average(sorted(vals)[:int(ratio*len(vals))])


@gob.mapper(all_items=True)
def eval_preds(preds):
    for key,vals in preds:
        local = sum(1 for v in vals if v<25)
        results = dict(
            local = local,
            per = 1.0*local/len(vals),
            aed60 = _aed(.6,vals),
            aed80 = _aed(.8,vals),
            aed100 = _aed(1,vals),
        )
        yield key,results

@gob.mapper(all_items=True)
def eval_stats(stats):
    for eval_key,groups in stats:
        print eval_key
        for stat_key in sorted(groups[0]):
            vals = [group[stat_key] for group in groups]
            print "%s\t%.3f\t%.5f"%(stat_key,np.average(vals),np.std(vals))


