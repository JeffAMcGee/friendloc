import itertools
import os
import errno
import sys
import math
import functools
from collections import defaultdict

import numpy as np

try:
    import simplejson as json
except:
    import json

from settings import settings
from models import *
from maroon import MongoDB, Model


def all_users():
    return User.get_all()


def grouper(n, iterable, fillvalue=None, dontfill=False):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    if dontfill:
        sentinel = object()
        fillvalue = sentinel
    args = [iter(iterable)] * n
    res = itertools.izip_longest(*args, fillvalue=fillvalue)
    if dontfill:
        f = functools.partial(itertools.ifilterfalse,lambda x: x is sentinel)
        res = itertools.imap(f,res)
    return res


def use_mongo(name):
    Model.database = mongo(name)


def mongo(name):
    return MongoDB(name=name,host=settings.db_host,slave_okay=True)


def in_local_box(place):
    box = settings.local_box
    return all(box[d][0]<place[d]<box[d][1] for d in ('lat','lng'))

def tri_users_dict_set(users_path):
    users = dict((int(d['id']),d) for d in _read_json(users_path))
    return users,set(users)


def read_gis_locs(path=None):
    for u in _read_json(path or "hou_tri_users"):
        yield u['lng'],u['lat']


def noisy(ray,scale):
    return ray+np.random.normal(0.0,scale,len(ray))

def median_2d(spots):
    return [np.median(x) for x in zip(*spots)]

def triangle_set(strict=True):
    Model.database = connect('houtx_user')
    users = Model.database.paged_view('_all_docs',include_docs=True,endkey="_")
    for row in users:
        user = row['doc']
        if user['prot'] or user['prob']==.5:
            continue
        if user['frdc']>2000 and user['folc']>2000:
            continue
        if strict and (user['prob']==0 or user['gnp'].get('pop',0)>1000000):
            continue
        yield user


def parse_ats(ats_path):
    ats =defaultdict(lambda: defaultdict(int))
    ated =defaultdict(lambda: defaultdict(int))
    for line in open(ats_path):
        uid,at = [int(i) for i in line.strip().split('\t')]
        ats[uid][at]+=1
        ated[at][uid]+=1
    return ats,ated

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as ex:
        if ex.errno!=errno.EEXIST:
            raise

def mainstream_edges(edges):
    return [ e for e in edges
        if 47<=e['lmfrd']+e['lmfol']<=300
        if 101<=e['lyfrd']+e['lyfol']<=954
        ]


def split_tri_counts(counts_path):
    edges = list(mainstream_edges(read_json(counts_path)))
    third = len(edges)/3
    return (edges[:third],edges[2*third:3*third],edges[third:2*third])


def _coord_delta(p1,p2):
    """estimate the distance between two points in miles - points can be 
    (lng,lat) tuple or dict(lng=lng,lat=lat)"""
    points = [
            (p['lng'],p['lat']) if isinstance(p, dict) else p
            for p in (p1,p2)]
    lng,lat=zip(*points)
    mlat = 69.1
    mlng = mlat*math.cos(sum(lat)*math.pi/360)
    return (mlat*(lat[0]-lat[1]), mlng*(lng[0]-lng[1]))


def coord_angle(p, p1, p2):
    "find the angle between rays from p to p1 and p2, return None if p in (p1,p2)"
    vs = [_coord_delta(p,x) for x in (p1,p2)]
    mags = [np.linalg.norm(v) for v in vs]
    if any(m==0 for m in mags):
        return math.pi
    cos = np.dot(*vs)/mags[0]/mags[1]
    return math.acos(min(cos,1))*180/math.pi


def coord_in_miles(p1, p2):
    """
    calculate the great circle distance using the haversine formula
    Taken from stackoverflow:
    http://stackoverflow.com/questions/4913349/
    """
    points = itertools.chain.from_iterable(
            (p['lng'],p['lat']) if isinstance(p, dict) else p
            for p in (p1,p2))
    lon1, lat1, lon2, lat2 = map(math.radians, points)
    # haversine formula 
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a)) 
    return 3959 * c


def np_haversine(lng1, lng2, lat1, lat2):
    """
    takes four numpy arrays in degrees and returns an array of the distances
    in miles.
    """
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    lng1r, lng2r = np.radians(lng1), np.radians(lng2)
    dlng, dlat = (lng2r - lng1r), (lat2r - lat1r)
    a = np.sin(dlat/2)**2 + np.cos(lat1r)*np.cos(lat2r)*np.sin(dlng/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return 3959 * c


def dist_bins(per_decade=10,start_exp=0,end_exp=5):
    space = np.linspace(start_exp,end_exp,1+(end_exp-start_exp)*per_decade)
    return np.insert(10**space,0,0)


def contact_prob(miles):
    # these numbers were determined by contact_fit
    return .008/(miles+2.094)


def read_json(path=None):
    file = open(path) if path else sys.stdin
    return (json.loads(l) for l in file)

def write_json(it, path=None):
    file = open(path,'w') if path else sys.stdout
    for d in it:
        print>>file, json.dumps(d)
    file.close()

def peek(iterable):
    it = iter(iterable)
    first = it.next()
    return first, itertools.chain([first],it)
