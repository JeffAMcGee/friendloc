import bisect
import math
import logging
import operator
import collections
from itertools import chain

import numpy as np
from scipy import stats, optimize

#from base.gisgraphy import GisgraphyResource
from base.models import User, Tweets
from base.utils import coord_in_miles
from base import gob, utils
from predict import fl


def _tile(deg):
    return int(math.floor(10*deg))


def _paged_users(uids, **find_kwargs):
    # save some round trips by asking for 100 at a time
    groups = utils.grouper(100, uids, dontfill=True)
    return chain.from_iterable(
        User.find(User._id.is_in(list(group)), **find_kwargs)
        for group in groups
    )


@gob.mapper(all_items=True)
def contact_count(uids):
    counts = collections.defaultdict(int)
    for contact in _paged_users(uids,fields=['gnp']):
        if contact.geonames_place and contact.geonames_place.mdist<1000:
            lat = _tile(contact.geonames_place.lat)
            lng = _tile(contact.geonames_place.lng)
            counts[lng,lat]+=1
    return counts.iteritems()


@gob.mapper(all_items=True)
def contact_mdist(uids):
    for contact in _paged_users(uids,fields=['gnp']):
        yield contact.geonames_place.mdist if contact.geonames_place else None


@gob.mapper(all_items=True)
def mloc_tile(mloc_uids):
    users = User.find(User._id.is_in(tuple(mloc_uids)),fields=['mloc','nebrs'])
    for user in users:
        if not user.neighbors:
            continue
        lng,lat = user.median_loc
        yield _tile(lat),user.to_d()


@gob.mapper()
def nebr_dists(mloc_tile):
    nebrs = User.find(User._id.is_in(mloc_tile['nebrs']),fields=['gnp'])
    for nebr in nebrs:
        dist = coord_in_miles(mloc_tile['mloc'], nebr.geonames_place.to_d())
        # add a one at the end to make the output format identical to
        # stranger_dists.
        yield dist,1


@gob.mapper(all_items=True)
def tile_split(groups):
    # FIXME this is evil boilerplate!
    for group,tiled_users in groups:
        for tiled_user in tiled_users:
            yield group,tiled_user


class StrangerDists(object):
    def __init__(self,env):
        self.env = env
        self.contact_count = dict(self.env.load('contact_count','mp'))
        self.contact_mat = None

    def _dists_for_lat(self,lat):
        lat_range = np.linspace(-89.95,89.95,1800)
        lng_range = np.linspace(.05,180.05,1801)
        lat_grid,lng_grid = np.meshgrid(lat_range, lng_range)

        centered_lat = .05 + .1*_tile(lat)
        lat_ar = np.empty_like(lat_grid)
        lat_ar.fill(centered_lat)
        lng_0 = np.empty_like(lat_grid)
        lng_0.fill(.05)

        return utils.np_haversine(lng_0, lng_grid, lat_ar, lat_grid)

    @gob.mapper(all_items=True)
    def stranger_dists(self, mloc_tile):
        mlocs = [m['mloc'] for m in mloc_tile]
        lat = mlocs[0][1]
        assert all(_tile(m[1])==_tile(lat) for m in mlocs)

        lngs = collections.defaultdict(int)
        for mloc in mlocs:
            lngs[_tile(mloc[0])]+=1

        dists = self._dists_for_lat(lat)
        for lng_tile, m_count in lngs.iteritems():
            for spot, c_count in self.contact_count.iteritems():
                c_lng,c_lat = spot
                if not ( -1800<=c_lng<1800 and -900<=c_lat<900):
                    continue
                delta = abs(lng_tile-c_lng)
                dist = dists[delta if delta<1800 else 3600-delta,c_lat+900]
                yield dist,m_count*c_count

    def _contact_mat(self):
        if self.contact_mat is not None:
            return self.contact_mat
        mat = np.zeros((3600,1800))
        for spot, c_count in self.contact_count.iteritems():
            (lng_tile,lat_tile) = spot
            if -900<=lat_tile<900:
                mat[(lng_tile+1800)%3600,lat_tile+900]+= c_count
        self.contact_mat = mat
        return mat

    @gob.mapper()
    def stranger_prob(self,lat_tile):
        lat_range = np.linspace(-89.95,89.95,1800)
        lng_range = np.linspace(.05,359.95,3600)
        lat_grid,lng_grid = np.meshgrid(lat_range, lng_range)

        dists = utils.np_haversine(.05, lng_grid, .1*lat_tile+.05, lat_grid)
        contact_mat = self._contact_mat()
        dists[0,lat_tile+900] = 2

        for lng_tile in xrange(-1800,1800):
            probs = np.log(1-utils.contact_prob(dists))
            prob = np.sum(contact_mat*probs)
            yield (lng_tile,lat_tile),prob
            dists = np.roll(dists,1,0)


@gob.mapper(all_items=True)
def stranger_mat(spots):
    mat = np.zeros((3600,1800),np.float32)
    for lng_lat,val in spots:
        if not np.isnan(val):
            mat[lng_lat[0]+1800,lng_lat[1]+900] = val
    yield mat


@gob.mapper()
def lat_tile():
    for tile in xrange(-900,900):
        yield abs(tile)//10,tile


@gob.mapper(all_items=True)
def strange_nebr_bins(dist_counts):
    counts = collections.defaultdict(int)
    bins = utils.dist_bins(120)
    for dist, count in dist_counts:
        bin = bisect.bisect(bins,dist)
        counts[bin]+=count
    return counts.iteritems()


def contact_curve(lm, a, b, c):
    return a*np.power(lm+b,c)


class ContactFit(object):
    def __init__(self,env):
        self.env = env

    def _counts(self, tups):
        # take tuples from nebr_bins or strange_bins and return np array of
        # counts
        bin_d = dict(tups)
        counts = [bin_d.get(b,0) for b in xrange(2,482)]
        return np.array(counts)

    def _bins(self):
        return utils.dist_bins(120)

    def _miles(self):
        bins = self._bins()
        # find the geometric mean of the non-zero bins
        return np.sqrt([bins[x-1]*bins[x] for x in xrange(2,482)])

    def _fit_stgrs(self, miles):
        # estimate the number of strangers at a distance from data >10 miles
        stgrs = self._counts(self.env.load('strange_bins'))
        sol = stats.linregress(np.log10(miles[120:]), np.log10(stgrs[120:]))
        return 10**(sol[0]*(np.log10(miles))+sol[1])

    @gob.mapper()
    def contact_fit(self):
        nebrs = self._counts(self.env.load('nebr_bins'))
        miles = self._miles()
        fit_stgrs = self._fit_stgrs(miles)
        ratios = nebrs/fit_stgrs

        # fit the porportion of strangers who are contacts to a curve.
        def curve(lm, a, b):
            return a/(lm+b)
        popt,pcov = optimize.curve_fit(curve,miles,ratios,(.01,3))
        print popt
        yield tuple(popt)

    @gob.mapper(all_items=True)
    def vect_fit(self,vects,in_paths):
        CHUNKS = 10
        bins = self._bins()
        miles = self._miles()
        fit_stgrs_ = self._fit_stgrs(miles)
        # FIXME: strange_bins was created from the whole dataset, but vects is
        # only based on the training set. This is fragile.
        fit_stgrs = .8*fit_stgrs_

        #load and classify the vects
        X, y = fl.vects_as_mat(vects)
        clump = in_paths[0][-1]
        nebr_clf = next(self.env.load('nebr_clf.'+clump,'pkl'))
        preds = nebr_clf.predict(X)

        # sort (predicted, actual) tuples by predicted value
        tups = zip(preds,y)
        tups.sort(key=operator.itemgetter(0))

        # unlogify the data from vects and break into chunks
        dists = np.power(2,[tup[1] for tup in tups])-.01
        splits = [len(tups)*x//CHUNKS for x in xrange(1,CHUNKS)]

        cutoff = 0
        for index,chunk in enumerate(np.split(dists,splits)):
            hist,b = np.histogram(chunk,bins)
            ratio = hist[1:481]/fit_stgrs

            popt,pcov = optimize.curve_fit(
                            contact_curve,
                            miles,
                            ratio,
                            (.001,2,-1),
                            1/miles,
                            ftol=.0001,
                            )
            print (cutoff,tuple(popt))
            yield (cutoff,tuple(popt))
            # FIXME: I think this is wrong -- should we use index+1 ?
            cutoff = tups[len(tups)*index//CHUNKS][0]


@gob.mapper()
def contact_blur(nebr_id):
    leafs = {}
    user = User.get_id(nebr_id)
    if user.local_friends is not None or user.local_followers is not None:
        return
    for key in ['rfriends','just_followers','just_friends']:
        cids = getattr(user,key)
        if cids:
            contacts = User.find(User._id.is_in(cids), fields=['gnp'])
            leafs[key] = [u for u in contacts if u.has_place()]
        else:
            leafs[key] = []
    user_loc = user.geonames_place.to_d()
    for kind in ('friends','followers'):
        contacts = leafs['rfriends'] + leafs['just_'+kind]
        dists = [
            coord_in_miles(user_loc,contact.geonames_place.to_d())
            for contact in contacts]
        if dists:
            blur = sum(1.0 for d in dists if d<25)/len(dists)
            yield len(dists),blur
        else:
            blur = None
            logging.info('no %s for %s - %d',kind,user.screen_name,user._id)
        setattr(user,'local_'+kind,blur)
    user.save()


@gob.mapper()
def geo_ats():
    for tweets in Tweets.find({},fields=['ats']):
        if tweets.ats:
            uid = tweets._id
            yield User.mod_id(uid), (uid,tweets.ats)


@gob.mapper()
def edges_d(user_d):
    me = User(user_d)
    if not me.neighbors:
        return []
    nebrs = set(me.neighbors)
    tweets = Tweets.get_id(me._id,fields=['ats'])
    ats = set(tweets.ats or [])

    #store neighbors users
    keys = {'just_followers':'jfol',
            'just_friends':'jfrd',
            'rfriends':'rfrd',
            'just_mentioned':'jat'}
    rels = dict(_id = me._id, mloc = me.median_loc)
    for long,short in keys.iteritems():
        amigos = [a for a in getattr(me,long) if a in nebrs]
        if amigos:
            rels[short] = _rel_d(User.get_id(amigos[0]),ats)

    return [rels]


def _rel_d(user,ats):
    gnp = user.geonames_place.to_d()
    return dict(
        folc=user.followers_count,
        frdc=user.friends_count,
        prot=user.protected,
        lat=gnp['lat'],
        lng=gnp['lng'],
        mdist=gnp['mdist'],
        ated=user._id in ats,
        _id=user._id,
        )


@gob.mapper()
def edge_dists(edge_d):
    keys = ('jfol','jfrd','rfrd','jat')
    for key in keys:
        amigo = edge_d.get(key)
        if amigo:
            assert amigo['mdist']<1000
            dist = coord_in_miles(edge_d['mloc'],amigo)
            yield (key,amigo['ated'],amigo['prot']),dist


@gob.mapper()
def edge_leaf_dists(edge_d):
    #FIXME: limit to contacts in 20..29 for now
    if not edge_d.get('rfrd') or str(edge_d['rfrd'])[-2]!='2':
        return
    rfrd = User.get_id(edge_d['rfrd']['_id'])
    edge_dist = coord_in_miles(edge_d['mloc'],edge_d['rfrd'])
    contacts = User.find(User._id.is_in(rfrd.contacts), fields=['gnp'])
    dists = [
        coord_in_miles(edge_d['rfrd'],contact.geonames_place.to_d())
        for contact in contacts
        if contact.has_place()
        ]
    if dists:
        yield edge_dist,dists
