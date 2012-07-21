import itertools
import calendar
import bisect
import os
import sys
import random
import math
import json
import logging
import collections
from datetime import datetime as dt
from datetime import timedelta
from operator import itemgetter
from multiprocessing import Pool

import numpy as np
from scipy import stats, optimize

from settings import settings
#from base.gisgraphy import GisgraphyResource
from base.models import User, Tweets, Edges, Tweet
from base.utils import coord_in_miles, use_mongo, write_json
from base import gob, utils


def _tile(deg):
    return int(math.floor(10*deg))

@gob.mapper(all_items=True)
def contact_count(uids):
    counts = collections.defaultdict(int)
    # save some round trips by asking for 100 at a time
    groups = utils.grouper(100, uids, dontfill=True)
    for group in groups:
        contacts = User.find(User._id.is_in(list(group)), fields=['gnp'])
        for contact in contacts:
            if contact.geonames_place and contact.geonames_place.mdist<1000:
                lat = _tile(contact.geonames_place.lat)
                lng = _tile(contact.geonames_place.lng)
                counts[lng,lat]+=1
    return counts.iteritems()


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

    def _dists_for_lat(self,lat):
        lat_range = np.linspace(-89.95,89.95,1800)
        lng_range = np.linspace(.05,180.05,1801)
        lat_grid,lng_grid = np.meshgrid(lat_range, lng_range)

        centered_lat = .05 + .1*_tile(lat)
        lat_ar = np.empty_like(lat_grid)
        lat_ar.fill(math.radians(centered_lat))
        lng_0 = np.empty_like(lat_grid)
        lng_0.fill(math.radians(.05))

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


@gob.mapper(all_items=True)
def strange_nebr_bins(dist_counts):
    counts = collections.defaultdict(int)
    bins = utils.dist_bins(120)
    for dist, count in dist_counts:
        bin = bisect.bisect(bins,dist)
        counts[bin]+=count
    return counts.iteritems()


class ContactFit(object):
    def __init__(self,env):
        self.env = env

    def _counts(self, tups):
        # take tuples from nebr_bins or strange_bins and return np array of
        # counts
        bin_d = dict(tups)
        counts = [bin_d.get(b,0) for b in xrange(2,482)]
        return np.array(counts)

    @gob.mapper()
    def contact_fit(self):
        nebrs = self._counts(self.env.load('nebr_bins'))
        stgrs = self._counts(self.env.load('strange_bins'))

        # find the geometric mean of the non-zero bins
        bins = utils.dist_bins(120)
        miles = np.sqrt([bins[x-1]*bins[x] for x in xrange(2,482)])

        # estimate the number of strangers at a distance from data >10 miles
        sol = stats.linregress(np.log10(miles[120:]), np.log10(stgrs[120:]))
        fit_stgrs = 10**(sol[0]*(np.log10(miles))+sol[1])
        ratios = nebrs/fit_stgrs

        # fit the porportion of strangers who are contacts to a curve.
        def curve(lm, a, b):
            return a/(lm+b)
        popt,pcov = optimize.curve_fit(curve,miles,ratios,(.01,3))
        print popt
        yield tuple(popt)


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
            yield dict(uid=tweets._id,ats=tweets.ats)


def find_geo_ats():
    "create geo_ats.json"
    for tweets in Tweets.find({},fields=['ats']):
        if tweets.ats:
            print json.dumps(dict(uid=tweets._id,ats=tweets.ats))


def crowdy_export(year, month, day):
    "export the tweets for crowd detection"
    uids = set(u['_id'] for u in User.coll().find({'ncd':{'$ne':None}}, fields=[]))
    for hour in xrange(24):
        start = dt(int(year), int(month), int(day), hour)
        end = start + timedelta(hours=1)
        tweets = Tweet.find(
            Tweet.created_at.range(start,end),
            sort='ca',
            fields=['ca','ats','uid'],
            )
        path = os.path.join(year,month,day,str(hour))
        mkdir_p(os.path.dirname(path))
        with open(path,'w') as f:
            for t in tweets:
                ts = int(calendar.timegm(t.created_at.timetuple()))
                for at in t.mentions:
                    if at in uids:
                        print>>f,"%d %d %d %d"%(ts,t._id,t.user_id,at)


def find_ats(users_path="hou_tri_users"):
    users, uids = tri_users_dict_set(users_path)
    for line in sys.stdin:
        d = json.loads(line)
        if d.get('ats'):
            local = int(d['uid']) in uids
            for at in d['ats']:
                if local or int(at) in uids:
                    print "%s\t%s"%(d['uid'],at)


def print_tri_counts():
    use_mongo('usgeo')
    pool = Pool(8,use_mongo,['usgeo'])
    users = list(User.find_connected(where="this._id % 10 <=6"))
    print 'loaded users'

    for key in ['rfrd','jat','fol','frd']:
        print 'working on '+key
        data = pool.map(
            _tri_for_user,
            ((user,key) for user in users),
            chunksize=100,
            )
        cleaned = sorted(itertools.ifilter(None, data), key=itemgetter('dist'))
        write_json(cleaned, 'geo_tri_%s'%key)
        print 'saved file'

def _tri_for_user(user_key):
    user, key = user_key
    keys = {'rfrd':'rfriends',
        'jat':'just_mentioned',
        'fol':'just_followers',
        'frd':'just_friends'
        }

    field = keys[key]
    field_ids = getattr(user,field)
    if not field_ids:
        return None

    your_id = field_ids[0]
    me = Edges.get_id(user._id)
    you = Edges.get_id(your_id)
    your_profile = User.get_id(your_id, fields=['gnp'])
    your_gnp = your_profile.geonames_place.to_d()

    mfrd = set(me.friends)
    mfol = set(me.followers)
    yfrd = set(you.friends)
    yfol = set(you.followers)
    all = (mfrd|mfol) & (yfrd|yfol)
    sets = dict(mfrd=mfrd, mfol=mfol, yfrd=yfrd, yfol=yfol)

    dist = coord_in_miles(user.median_loc,your_gnp)
    d = dict(
            dist=dist,
            all=len(all),
            uid=user._id,
            aid=your_id,
            mdist=your_gnp['mdist'],
            )
    for k,v in sets.iteritems():
        d['l'+k]= len(v)
        d[k] = list(all&v)
    return d
       
def startup_usgeo():
    TwitterModel.database = mongo('usgeo')


def save_user_json(func,debug=False):
    startup_usgeo()
    if debug:
        settings.pdb()
    users = User.find(
            User.median_loc.exists(),
            timeout=False,
            where="this._id % 10 <=6")
    if debug:
        _map, f = itertools.imap, sys.stdout
    else:
        p = Pool(8,initializer=startup_usgeo)
        _map, f = p.imap_unordered, open(func,'w')
    for item in _map(globals()[func],users):
        if item:
            print>>f, json.dumps(item)


def rfr_triads(me):
    #./do.py save_user_json rfr_triads
    me_rfr = set(me.rfriends or []).intersection(me.neighbors or [])
    if len(me_rfr)<3:
        return None
    for you_id in me_rfr:
        you_ed = Edges.get_id(you_id)
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
            return d
    return None


def rfr_net(me):
    #./do.py save_user_json rfr_net 
    me_rfr = set(me.rfriends).intersection(me.neighbors)
    rfrs = []
    for you_id in me_rfr:
        edges = Edges.get_id(you_id,fields=['fols'])
        gnp = User.get_id(you_id,fields=['gnp']).geonames_place.to_d()
        rfrs.append(dict(
            lat=gnp['lat'],
            lng=gnp['lng'],
            mdist=gnp['mdist'],
            _id=you_id,
            fols=list(me_rfr.intersection(edges.followers)),
            ))
    return dict(
        _id=me._id,
        mloc = me.median_loc,
        rfrs = rfrs,
        )

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
    if 'mdist' not in gnp:
        gnp['mdist'] = gis.mdist(user.geonames_place)
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


