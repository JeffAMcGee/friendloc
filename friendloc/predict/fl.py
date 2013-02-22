import math
from itertools import chain
import bisect
from collections import defaultdict, Counter

from sklearn import tree
import numpy as np

from friendloc.base import gob, utils
from friendloc.base.utils import coord_in_miles
from friendloc.predict import prep
import friendloc.explore.peek


def logify(x,fudge=1):
    return math.log(x+fudge,2)


def unlogify(x,fudge=1):
    return 2**x - fudge


def _load_geo_ated(geo_ated):
    return {to:set(froms) for to, froms in geo_ated}


@gob.mapper(slurp={'geo_ated':_load_geo_ated,'dirt_cheap_locals':dict})
def nebr_vect(user,geo_ated,dirt_cheap_locals):
    """
    create a vector for each edge from a target to a contact
    """
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
        # why not nan for missing lorat?
        lorat = dirt_cheap_locals.get(nebr['_id'],.25)
        others = [ lorat, int(bool(nebr['prot'])), mloc_dist, ]
        yield flags + logged + others


def vects_as_mat(vects):
    """
    convert nebr_vects to a scaled numpy array
    """
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
    """
    train two classifiers from nebr_vect data: one using information from leafs,
    and one without.
    """
    X, y = vects_as_mat(vects)
    clf = tree.DecisionTreeRegressor(min_samples_leaf=1000)
    clf.fit(X,y)

    # We want to ignore columns from X that depend on the leafs and crawling the
    # tweets of the contacts.
    #FIXME: the set of columns we are ignoring is fragile
    for col in (1,2,9):
        X[:,col] = 0
    near_clf = tree.DecisionTreeRegressor(min_samples_leaf=1000)
    near_clf.fit(X,y)
    yield dict(leaf=clf,contact=near_clf)


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
        return np.argmin(nebrs_d['pred_dists']['leaf'])

class Last(Predictor):
    def predict(self,nebrs_d,vect_fit):
        return len(nebrs_d['strange_prob'])-1

class FacebookMLE(Predictor):
    def predict(self,nebrs_d,vect_fit):
        contact_prob = nebrs_d['contact_prob']
        probs = np.sum(np.log(contact_prob/(1-contact_prob)),axis=0)
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


class FriendLoc(Predictor):
    def __init__(self, env, loc_factor=0, tz_factor=0, strange_factor=0,
                 force_loc=False, clf_key='leaf'):
        self.env = env
        self.loc_factor = loc_factor
        self.strange_factor = strange_factor
        self.tz_factor = tz_factor
        self.force_loc = force_loc
        self.clf_key = clf_key

    def predict(self,nebrs_d,vect_fit):
        cutoffs = [cut for k,cut,cur in vect_fit if k==self.clf_key]
        curves = [cur for k,cut,cur in vect_fit if k==self.clf_key]

        if self.force_loc and nebrs_d['gnp'] and nebrs_d['gnp']['mdist']<25:
            return len(nebrs_d['vects'])
        mat = nebrs_d['dist_mat']
        probs = np.empty_like(mat)
        for index,pred in enumerate(nebrs_d['pred_dists'][self.clf_key]):
            curve = curves[bisect.bisect(cutoffs,pred)-1]
            probs[index,:] = friendloc.explore.peek.contact_curve(mat[index,:],*curve)
        contact_prob = nebrs_d['contact_prob']
        total_probs = (
            np.sum(np.log(probs/(1-contact_prob)),axis=0) +
            self.strange_factor*nebrs_d['strange_prob'] +
            self.tz_factor*nebrs_d['tz_prob'] +
            self.loc_factor*nebrs_d['location_prob']
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
    """
    predict location for target users using a variety of methods, including a
    few baselines
    """
    # This function is a stepping-stone to removing a useless class.
    p = Predictors(env)
    return p.predictions(nebrs_ds,in_paths,geo_ated,dirt_cheap_locals)


class Predictors(object):
    def __init__(self,env):
        self.env = env
        self.classifiers = dict(
            omni=Omni(),
            nearest=Nearest(),
            backstrom=FacebookMLE(),
            friendloc_nostrange=FriendLoc(env,0),
            friendloc_loc=FriendLoc(env,loc_factor=1,strange_factor=1),
            friendloc_tz=FriendLoc(env,tz_factor=1,strange_factor=1),
            friendloc_basic=FriendLoc(env,strange_factor=1),
            friendloc_near=FriendLoc(env,strange_factor=1,clf_key='contact'),
            friendloc_cut=FriendLoc(env,strange_factor=1,force_loc=True),
            friendloc_cutloc=FriendLoc(env,strange_factor=1,loc_factor=1,force_loc=True),
            friendloc_nearcut=FriendLoc(env,strange_factor=1,clf_key='contact',force_loc=True),
        )

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
        lng_t = friendloc.explore.peek._tile(nebr['lng'])
        lat_t_ = friendloc.explore.peek._tile(nebr['lat'])
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

        nebrs_d['pred_dists'] = {
                k:clf.predict(X) for k,clf in
                self.nebr_clf.iteritems()
            }
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

    def predict(self, user, nebrs, ats, ated, clf='friendloc_cut'):
        # this is an ugly way to deal with geo_ated and cheap_locals
        self.geo_ated = {user._id:ated}

        def real_or_none(x):
            return None if x is None or math.isnan(x) else x

        self.cheap_locals = {
            nebr._id : real_or_none(nebr.local_ratio)
            for nebr in nebrs
        }

        nebrs_d = prep.make_nebrs_d(user,nebrs,ats)
        self.prep_nebrs(nebrs_d)

        index = self.classifiers[clf].predict(nebrs_d,self.vect_fit)
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
    """
    evaluate the predictions using average error distance and accuracy
    """
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
    """
    combine the results from the five folds to get average and standard
    deviation of the results
    """
    for eval_key,groups in sorted(stats):
        print eval_key
        row = []
        for stat_key in ('aed60','aed80','aed100','per'):
            vals = [group[stat_key] for group in groups]
            row.append(np.average(vals))
            row.append(np.std(vals))
        row[6]*=100
        row[7]*=100
        line = "%.2f$\\pm$%.3f & %.1f$\\pm$%.2f & %.3g$\\pm$%.1f & %.1f\\%%$\\pm$%.2f\\%% \\\\"
        print (line%tuple(row)).replace('pm$0.','pm$.')

