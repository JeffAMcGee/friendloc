import itertools
import time
import os
import sys
import random
import logging
from collections import defaultdict
from datetime import datetime as dt
from datetime import timedelta
from operator import itemgetter
from multiprocessing import Pool

from maroon import ModelCache

from settings import settings
import base.twitter as twitter
from base.gisgraphy import GisgraphyResource
from base.splitproc import SplitProcess
from base.models import *
from base.utils import *


gis = GisgraphyResource()


def find_geo_ats():
    "create geo_ats.json"
    for tweets in Tweets.find({},fields=['ats']):
        if tweets.ats:
            print json.dumps(dict(uid=tweets._id,ats=tweets.ats))


def crowdy_export(year, month, day):
    "export the tweets for crowd detection"
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
                ts = int(time.mktime(t.created_at.timetuple()))
                for at in t.mentions:
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

def edges_json(me):
    #./do.py save_user_json edges_json 
    tweets = Tweets.get_id(me._id,fields=['ats'])
    ats = set(tweets.ats or [])

    #store public users
    keys = {'just_followers':'jfol',
            'just_friends':'jfrd',
            'rfriends':'rfrd',
            'just_mentioned':'jat'}
    rels = dict(_id = me._id, mloc = me.median_loc)
    for long,short in keys.iteritems():
        amigos = getattr(me,long)
        if amigos:
            rels[short] = _rel_d(User.get_id(amigos[0]),ats)

    #get a list of all the user's relationships
    edges = Edges.get_id(me._id)
    frds = set(edges.friends or [])
    fols = set(edges.followers or [])
    lookups = edges.lookups if edges.lookups else list(ats|frds|fols)
    
    #get the protected users - this will be SLOW
    pusers = User.find(
            User._id.is_in(lookups) & (User.protected==True),
            fields =['gnp','frdc','folc']
            )
    puser_groups = defaultdict(list)
    #pick random protected users
    for puser in pusers:
        if puser._id in frds:
            label = 'prfrd' if puser._id in fols else 'pjfrd'
        else:
            label = 'pjfol' if puser._id in fols else 'pjat'
        puser_groups[label].append(puser)
    for label,users in puser_groups.iteritems():
        rels[label] = _rel_d(random.choice(users),ats)
    return rels

def _rel_d(user,ats):
    gnp = user.geonames_place.to_d()
    if 'mdist' not in gnp:
        gnp['mdist'] = gis.mdist(user.geonames_place)
    return dict(
        folc=user.followers_count,
        frdc=user.friends_count,
        lat=gnp['lat'],
        lng=gnp['lng'],
        mdist=gnp['mdist'],
        ated=user._id in ats,
        _id=user._id,
        )
