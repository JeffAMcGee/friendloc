import bisect
import math
import logging
import operator
import collections
import random
from itertools import chain

import numpy as np
from scipy import stats, optimize

from friendloc.base.models import User, Tweets, Edges
from friendloc.base.utils import coord_in_miles
from friendloc.base import gob, utils
import friendloc.predict.fl

def local_ratio(dists,cutoff=25):
    return sum(1.0 for d in dists if d<cutoff)/len(dists)


def _tile(deg):
    """break earth into .1x.1 degree tiles"""
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
    """
    count the number of contacts in each .1x.1 degree tile
    """
    counts = collections.defaultdict(int)
    for contact in _paged_users(uids,fields=['gnp']):
        if contact.geonames_place and contact.geonames_place.mdist<1000:
            lat = _tile(contact.geonames_place.lat)
            lng = _tile(contact.geonames_place.lng)
            counts[lng,lat]+=1
    return counts.iteritems()


@gob.mapper(all_items=True)
def contact_mdist(uids):
    """fetch median location error for a set of user ids"""
    for contact in _paged_users(uids,fields=['gnp']):
        yield contact.geonames_place.mdist if contact.geonames_place else None


@gob.mapper(all_items=True)
def diff_mloc_mdist(uids):
    """
    for each target user return the location error and median location error
    """
    for contact in _paged_users(uids,fields=['gnp','mloc']):
        if contact.geonames_place:
            dist = coord_in_miles(contact.geonames_place.to_d(),contact.median_loc)
            yield dist,contact.geonames_place.mdist


@gob.mapper(all_items=True)
def mlocs(uids):
    """
    fetch the home locations for all the target users
    """
    for target in _paged_users(uids,fields=['mloc']):
        yield target.median_loc


@gob.mapper(all_items=True)
def mloc_tile(mloc_uids):
    """
    split the target users into tiles based on their home location
    """
    users = User.find(User._id.is_in(tuple(mloc_uids)),fields=['mloc','nebrs'])
    for user in users:
        if not user.neighbors:
            continue
        lng,lat = user.median_loc
        yield _tile(lat),user.to_d()


@gob.mapper()
def nebr_dists(mloc_tile):
    """
    find the distances from target users to their contacts
    """
    nebrs = User.find(User._id.is_in(mloc_tile['nebrs']),fields=['gnp'])
    for nebr in nebrs:
        dist = coord_in_miles(mloc_tile['mloc'], nebr.geonames_place.to_d())
        # add a one at the end to make the output format identical to
        # stranger_dists.
        yield dist,1


@gob.mapper(all_items=True)
def tile_split(groups):
    """split tiles into seperate files"""
    # FIXME this is evil boilerplate!
    for group,tiled_users in groups:
        for tiled_user in tiled_users:
            yield group,tiled_user


def _dists_for_lat(lat):
    lat_range = np.linspace(-89.95,89.95,1800)
    lng_range = np.linspace(.05,180.05,1801)
    lat_grid,lng_grid = np.meshgrid(lat_range, lng_range)

    centered_lat = .05 + .1*_tile(lat)
    lat_ar = np.empty_like(lat_grid)
    lat_ar.fill(centered_lat)
    lng_0 = np.empty_like(lat_grid)
    lng_0.fill(.05)

    return utils.np_haversine(lng_0, lng_grid, lat_ar, lat_grid)


@gob.mapper(all_items=True,slurp={'contact_count':dict})
def stranger_dists(mloc_tile, contact_count):
    """
    find the distances from every target user to every contact (even between
    pairs with no relationship.)
    """
    mlocs = [m['mloc'] for m in mloc_tile]
    lat = mlocs[0][1]
    assert all(_tile(m[1])==_tile(lat) for m in mlocs)

    lngs = collections.defaultdict(int)
    for mloc in mlocs:
        lngs[_tile(mloc[0])]+=1

    dists = _dists_for_lat(lat)
    for lng_tile, m_count in lngs.iteritems():
        for spot, c_count in contact_count.iteritems():
            c_lng,c_lat = spot
            if not ( -1800<=c_lng<1800 and -900<=c_lat<900):
                continue
            delta = abs(lng_tile-c_lng)
            dist = dists[delta if delta<1800 else 3600-delta,c_lat+900]
            yield dist,m_count*c_count


def _contact_mat(contact_count):
    mat = np.zeros((3600,1800))
    for spot, c_count in contact_count:
        (lng_tile,lat_tile) = spot
        if -900<=lat_tile<900:
            mat[(lng_tile+1800)%3600,lat_tile+900]+= c_count
    return mat


@gob.mapper()
def lat_tile():
    """divide the earth into 180 slices to calculate stranger_prob """
    for tile in xrange(-900,900):
        yield abs(tile)//10,tile


@gob.mapper(slurp={'contact_count':_contact_mat})
def stranger_prob(lat_tile,contact_count):
    """
    Calculate pStrangers for every longitude tile at a specific latitude.
    pStrangers is the probability that a user lives at a location given the
    locations of people they are not connected to. This step of FreindlyLocation
    took about 2 weeks on a machine with 8 cores.
        lat_tile should be between -900 and 900 and represents a latitude
        contact_count is a matrix of the locations of the contacts
    """
    lat_range = np.linspace(-89.95,89.95,1800)
    lng_range = np.linspace(.05,359.95,3600)
    lat_grid,lng_grid = np.meshgrid(lat_range, lng_range)

    dists = utils.np_haversine(.05, lng_grid, .1*lat_tile+.05, lat_grid)
    # FIXME: the name of a slurped command-line argument should not have to
    # match the file name
    contact_mat = contact_count
    dists[0,lat_tile+900] = 2

    for lng_tile in xrange(-1800,1800):
        probs = np.log(1-utils.contact_prob(dists))
        prob = np.sum(contact_mat*probs)
        yield (lng_tile,lat_tile),prob
        dists = np.roll(dists,1,0)


@gob.mapper(all_items=True)
def stranger_mat(spots):
    """
    combine all the values from stranger_prob in a matrix
    """
    mat = np.zeros((3600,1800),np.float32)
    for lng_lat,val in spots:
        if not np.isnan(val):
            mat[lng_lat[0]+1800,lng_lat[1]+900] = val
    yield mat


@gob.mapper(all_items=True)
def strange_nebr_bins(dist_counts):
    """
    count the number of strangers or contacts at a specific distance from a
    target user
    """
    counts = collections.defaultdict(int)
    bins = utils.dist_bins(120)
    for dist, count in dist_counts:
        bin = bisect.bisect(bins,dist)
        counts[bin]+=count
    return counts.iteritems()


@gob.mapper(all_items=True,slurp={'mlocs':list})
def exact_strange_bins(uids,mlocs):
    """find the distance between every contact and every target user"""
    mlngs,mlats = np.transpose(mlocs)
    bins = utils.dist_bins(120)
    counts = np.zeros(len(bins)-1)
    for contact in _paged_users(set(uids),fields=['gnp']):
        clat = contact.geonames_place.lat
        clng = contact.geonames_place.lng
        dists= utils.np_haversine(clng, mlngs, clat, mlats)
        hist,b = np.histogram(dists,bins)
        counts+=hist
    return enumerate(counts)


def contact_curve(lm, a, b, c):
    return a*np.power(lm+b,c)


def _bin_counts(tups):
    # take tuples from nebr_bins or strange_bins and return np array of
    # counts
    bin_d = dict(tups)
    counts = [bin_d.get(b,0) for b in xrange(1,481)]
    return np.array(counts)


def _miles():
    bins = utils.dist_bins(120)
    # find the geometric mean of the non-zero bins
    return np.sqrt([bins[x-1]*bins[x] for x in xrange(2,482)])


def _fit_stgrs(miles, stgrs):
    # estimate the number of strangers at a distance from data >10 miles
    sol = stats.linregress(np.log10(miles[120:]), np.log10(stgrs[120:]))
    return 10**(sol[0]*(np.log10(miles))+sol[1])


@gob.mapper(slurp={'strange_bins':_bin_counts,'nebr_bins':_bin_counts})
def contact_fit(strange_bins,nebr_bins):
    """
    fit the porportion of strangers at a certain distance who are contacts to a
    curve.
    """
    miles = _miles()
    fit_stgrs = _fit_stgrs(miles,strange_bins)
    ratios = nebr_bins/fit_stgrs

    def curve(lm, a, b):
        return a/(lm+b)
    popt,pcov = optimize.curve_fit(curve,miles,ratios,(.01,3))
    print popt
    yield tuple(popt)


# FIXME: should vect_ratios go into predict.fl? I don't like this module
# depending on that module.
@gob.mapper(all_items=True,slurp={'exact_strange_bins':_bin_counts})
def vect_ratios(vects,in_paths,env,exact_strange_bins):
    """
    find the proportion of contacts who fit in a quantile
    """
    QUANTILES = 10
    bins = utils.dist_bins(120)
    # FIXME: exact_strange_bins was created from the whole dataset, but vects is
    # only based on the training set. This is fragile.
    strange_bins = .8*exact_strange_bins

    #load and classify the vects
    X, y = friendloc.predict.fl.vects_as_mat(vects)
    clump = in_paths[0][-1]
    # FIXME: Is there a nicer way to do this? We should be able to use two
    # inputs together.
    nebr_clf = next(env.load('nebr_clf.'+clump,'pkl'))
    for version in ('contact','leaf'):
        preds = nebr_clf[version].predict(X)

        # sort (predicted, actual) tuples by predicted value
        tups = zip(preds,y)
        tups.sort(key=operator.itemgetter(0))

        # unlogify the data from vects and break into quantiles
        dists = np.power(2,[tup[1] for tup in tups])-.01
        splits = [len(tups)*x//QUANTILES for x in xrange(1,QUANTILES)]

        for index,chunk in enumerate(np.split(dists,splits)):
            hist,b = np.histogram(chunk,bins)
            ratio = hist[1:481]/strange_bins
            cutoff = tups[len(tups)*index//QUANTILES][0]
            yield (version, cutoff, tuple(ratio))


@gob.mapper(all_items=True)
def vect_fit(vect_ratios):
    """
    fit the curve for the probability of being a contact (pContact)
    """
    miles = _miles()
    for version,cutoff,ratio in vect_ratios:
        popt,pcov = optimize.curve_fit(
                        contact_curve,
                        miles,
                        ratio,
                        (.001,2,-1),
                        miles**-.8,
                        ftol=.0001,
                        )
        print (version,cutoff,tuple(popt))
        yield (version,cutoff,tuple(popt))


@gob.mapper(all_items=True,slurp={'mloc_uids':set})
def dirt_cheap_locals(nebr_ids,mloc_uids):
    """
    local contact ratio based on 10 leafs
    """
    return cheap_locals(nebr_ids,mloc_uids,10)


@gob.mapper(all_items=True,slurp={'mloc_uids':set})
def aint_cheap_locals(nebr_ids,mloc_uids):
    """
    local contact ratio based on 100 leafs
    """
    return cheap_locals(nebr_ids,mloc_uids,100)


@gob.mapper(all_items=True,slurp={'mloc_uids':set})
def cheap_locals(nebr_ids,mloc_uids,cutoff=20):
    """
    local contact ratio based on 20 leafs
    """
    seen = set()
    # There can be duplicates because nebr_ids is created by clumping nebr_split
    for nebr_id in nebr_ids:
        if nebr_id in seen:
            continue
        seen.add(nebr_id)

        user = User.get_id(nebr_id)
        user_loc = user.geonames_place.to_d()

        cids = [
            cid
            for key in User.NEBR_KEYS
            for cid in (getattr(user,key) or [])
            if cid not in mloc_uids
            ]
        if not cids:
            continue
        random.shuffle(cids)
        leafs = User.find(User._id.is_in(cids[:cutoff]), fields=['gnp'])

        dists = [
            coord_in_miles(user_loc,leaf.geonames_place.to_d())
            for leaf in leafs
            if leaf.has_place()
        ]
        if dists:
            blur = sum(1.0 for d in dists if d<25)/len(dists)
            yield user._id,blur


@gob.mapper()
def geo_ats():
    """
    fetch all at mentions from database
    """
    for tweets in Tweets.find({},fields=['ats']):
        if tweets.ats:
            uid = tweets._id
            yield User.mod_id(uid), (uid,tweets.ats)


@gob.mapper()
def at_tuples(geo_at):
    """
    create (mentioned user, tweet creator) pairs from geo_ats, and split based
    on user id of mentioned user
    """
    uid,ats = geo_at
    for at in ats:
        yield User.mod_id(at), (at,uid)


@gob.mapper(all_items=True,slurp={'mloc_uids':set})
def geo_ated(at_tuples,mloc_uids):
    """
    create dictionary mapping target user ids to contacts who mentioned the
    taregt users
    """
    ated = collections.defaultdict(list)
    for to, frm in at_tuples:
        if to in mloc_uids:
            ated[to].append(frm)
    return ated.iteritems()


def _ated(ats,from_id,to_id):
    return from_id in ats and to_id in ats[from_id]


def _in_usa(lng,lat):
    return 24<lat<50 and -126<lng<-66


@gob.mapper(slurp={'geo_ats':dict})
def edges_d(user_d, geo_ats):
    """
    create one dict per target user with information about one selected contact
    for each of the four types of contact
    """
    me = User(user_d)
    if not me.neighbors:
        return []
    nebrs = set(me.neighbors)
    me_usa = _in_usa(me.median_loc[0],me.median_loc[1])

    keys = {'just_followers':'jfol',
            'just_friends':'jfrd',
            'rfriends':'rfrd',
            'just_mentioned':'jat'}
    rels = dict(_id = me._id, mloc = me.median_loc)
    for long,short in keys.iteritems():
        amigos = [a for a in getattr(me,long) if a in nebrs]
        if not amigos:
            continue
        amigo = User.get_id(amigos[0])
        gnp = amigo.geonames_place.to_d()
        if gnp['mdist']>1000:
            continue
        rels[short] = dict(
                folc=amigo.followers_count,
                frdc=amigo.friends_count,
                lofrd=amigo.local_friends,
                lofol=amigo.local_followers,
                prot=amigo.protected,
                lat=gnp['lat'],
                lng=gnp['lng'],
                mdist=gnp['mdist'],
                _id=amigo._id,
                i_at=_ated(geo_ats,me._id,amigo._id),
                u_at=_ated(geo_ats,amigo._id,me._id),
                usa = me_usa and _in_usa(gnp['lng'],gnp['lat']),
                )

    return [rels]


@gob.mapper()
def edge_dists(edge_d):
    """
    distill an edge_d into a smaller amount of information about each of the
    types of edges
    """
    keys = ('jfol','jfrd','rfrd','jat')
    for key in keys:
        amigo = edge_d.get(key)
        if amigo:
            assert amigo['mdist']<1000
            dist = coord_in_miles(edge_d['mloc'],amigo)
            yield (key,amigo['i_at'],amigo['u_at'],amigo['prot']),dist
            if key=='rfrd' and amigo['usa']:
                yield ('usa',amigo['i_at'],amigo['u_at'],amigo['prot']),dist


@gob.mapper(slurp={'dirt_cheap_locals':dict,'cheap_locals':dict,'aint_cheap_locals':dict})
def rfrd_dists(edge_d,dirt_cheap_locals,cheap_locals,aint_cheap_locals):
    """
    get the three different local contact ratios for recip friends, combine with
    actual distance from target to recip friend
    """
    amigo = edge_d.get('rfrd')
    if amigo:
        amigo['dist'] = coord_in_miles(edge_d['mloc'],amigo)
        amigo['cheap'] = cheap_locals.get(amigo['_id'])
        amigo['dirt'] = dirt_cheap_locals.get(amigo['_id'])
        amigo['aint'] = aint_cheap_locals.get(amigo['_id'])
        yield amigo


@gob.mapper()
def rfr_triads(user_d):
    """
    find a target users with a social triangle and a recip friend not in that
    triangle. Return info about all four users.
    """
    # We are looking for this structure in the social graph:
    # my  you---our
    #   \  |  /
    #      me
    # me is a target user, the other users are contacts, and the edges are all
    # reciprocal.
    me = User(user_d)
    me_rfr = set(me.rfriends or []).intersection(me.neighbors or [])
    if len(me_rfr)<3:
        return []
    for you_id in me_rfr:
        you_ed = Edges.get_id(you_id)
        if not you_ed:
            continue #There are no edges for this neighbor.
        ours = me_rfr.intersection(you_ed.friends,you_ed.followers)
        mine = me_rfr.difference(you_ed.friends,you_ed.followers)
        if ours and mine:
            d = dict(
                me = dict(_id=me._id,loc=me.median_loc),
                you = dict(_id=you_id),
                my = dict(_id=random.choice(list(mine))),
                our = dict(_id=random.choice(list(ours))),
                )
            for k,v in d.iteritems():
                if k=='me': continue
                gnp = User.get_id(v['_id'],fields=['gnp']).geonames_place.to_d()
                gnp.pop('zipcode',None)
                v['loc'] = gnp
            return [d]
    return []
