import itertools
import time
import os
import sys
import random
import logging
import math
import simplejson
from collections import defaultdict
from datetime import datetime as dt
from operator import itemgetter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import Bbox
import numpy

from settings import settings
import localcrawl.twitter as twitter
import localcrawl.gisgraphy as gisgraphy
from localcrawl.models import *
from maroon import ModelCache
from peek import _tri_users_dict_set


def _read_gis_locs(path=None):
    for u in _read_json(path or "hou_tri_users"):
        yield u['lng'],u['lat']

def _noisy(ray,scale):
    return ray+numpy.random.normal(0.0,scale,len(ray))

def plot_tweets():
    #usage: peek.py print_locs| peek.py plot_tweets
    locs = _read_gis_locs()
    mid_x,mid_y = (-95.4,29.8)
    box = settings.local_box
    lngs,lats = zip(*[
            c[0:2] for c in locs
            #if math.hypot(c[0]-mid_x,c[1]-mid_y)<10
            #if -96<c[0]<-94.6 and 29.2<c[1]<30.4
            ])
    fig = plt.figure(figsize=(18,18))
    ax = fig.add_subplot(111)
    cmap = LinearSegmentedColormap.from_list("gray_map",["#c0c0c0","k"])
    ax.plot(_noisy(lngs,.005),_noisy(lats,.005),',',
            color='k',
            alpha=.2,
            )
    print len(lngs)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    #ax.set_title("Tweets from Houston, TX (11/26/2010-1/14/2010)")
    fig.savefig('../www/hou_gis.png')


def dist_histogram():
    mid_x,mid_y = (-95.4,29.8)
    locs = read_locs()
    dists = [math.hypot((loc[0]-mid_x),loc[1]-mid_y) for loc in locs]
    dists = [x for x in dists if x<5]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(dists,bins=100)
    fig.savefig('../www/hist.png')


def tpu_hist(path=None):
    counts = defaultdict(int)
    for t in _read_json(path):
        counts[t['user']['id']]+=1
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    ax.hist(counts.values(),
            bins=xrange(1000),
            log=True,
            histtype='step',
            #cumulative=True,
            )
    ax.set_xscale('log')
    fig.savefig('../www/tpu_us_hist.png')


def diff_gnp_gps(path=None):
    gis = gisgraphy.GisgraphyResource()
    names = {}
    latlngs = defaultdict(list)
    for t in _read_json(path):
        uid = t['user']['id']
        if 'coordinates' not in t: continue
        lng,lat = t['coordinates']['coordinates']
        latlngs[uid].append(dict(lng=lng,lat=lat))
        names[uid] = t['user']['location']
    dists = []
    for uid,spots in latlngs.iteritems():
        if len(spots)<2: continue
        if _coord_in_miles(spots[0],spots[1])>30: continue
        gnp = gis.twitter_loc(names[uid])
        if not gnp: continue
        if gnp.population>5000000: continue
        dist = _coord_in_miles(gnp.to_d(),spots[0])
        dists.append(dist)
        if dist>100:
            d = gnp.to_d()
            d['uid'] = uid
            d['spot'] = spots[0]
            d['claim'] = names[uid]
            print simplejson.dumps(d)
    fig = plt.figure(figsize=(18,12))
    ax = fig.add_subplot(111)
    ax.hist(dists,
            bins=xrange(200), #[2**i for i in xrange(15)],
            #log=True,
            #normed =True,
            histtype='step',
            cumulative=True,
            )
    ax.set_xlabel('miles between location and tweet')
    ax.set_ylabel('number of users in bin')
    #ax.set_xscale('log')
    fig.savefig('../www/diff_gnp_gps.png')
    print len(dists)



def user_stddev(path=None):
    means = []
    locs = sorted(read_locs(path),key=itemgetter(2))
    for k,g in itertools.groupby(locs,key=itemgetter(2)):
        lats,lngs = zip(*[x[0:2] for x in g])
        if len(lats)==1: continue
        m_lat,m_lng = numpy.mean(lats),numpy.mean(lngs)
        dists = (math.hypot(m_lat-x,m_lng-y) for x,y in zip(lats,lngs))
        mean = sum(dists)/len(lats)
        #if mean<2:
        means.append(70*mean)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(means,bins=100,log=False)
    fig.savefig('../www/user.png')
    print numpy.median(means)



def _triangle_set(strict=True):
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


def graph_edges(users_path="hou_tri_users", ats_path="hou_ats"):
    users, uids = _tri_users_dict_set(users_path)
    ats,ated = _parse_ats(ats_path)
    counts = dict(non=[],frd=[],fol=[],rfrd=[],at=[],ated=[],conv=[])
    for u in users:
        obj = Edges.get_id(int(u))
        if not obj: continue
        sets = dict(
            frd = uids.intersection(obj.friends),
            fol = uids.intersection(obj.followers),
            )
        sets['rfrd'] = sets['frd'] & sets['fol']
        sets['non'] = uids.difference(sets['frd'] | sets['fol'])
        sets['at'] = ats[int(u)].keys()
        sets['ated'] = ated[int(u)].keys()
        sets['conv'] = set(sets['at']).intersection(sets['ated'])
        rfriends = len(sets['rfrd'])
        #if not rfriends or not sets['conv']: continue
        for k,v in sets.iteritems():
            if not v: continue
            others = random.sample(v,1)
            dists = [_coord_in_miles(users[u],users[other]) for other in others]
            counts[k].extend(dists)
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    for k,v in counts.iteritems():
        ax.hist(v,
                bins=numpy.arange(0,100,.2),
                histtype='step',
                label=k,
                #normed=True,
                cumulative=True)
    ax.set_ylim(0,15300)
    print len(counts['rfrd'])
    ax.legend()
    ax.set_xlabel('miles between users')
    ax.set_ylabel('count of users')
    fig.savefig('../www/edges.png')


def _parse_ats(ats_path):
    ats =defaultdict(lambda: defaultdict(int))
    ated =defaultdict(lambda: defaultdict(int))
    for line in open(ats_path):
        uid,at = [int(i) for i in line.strip().split('\t')]
        ats[uid][at]+=1
        ated[at][uid]+=1
    return ats,ated


def mainstream_edges(edges):
    return [ e for e in edges
        if 47<=e['lmfrd']+e['lmfol']<=300
        if 101<=e['lyfrd']+e['lyfol']<=954
        ]


def split_tri_counts(counts_path):
    edges = list(mainstream_edges(_read_json(counts_path)))
    third = len(edges)/3
    return (edges[:third],edges[2*third:3*third],edges[third:2*third])


def graph_thickness(counts_path="tri_counts"):
    far = 1
    fig = plt.figure(figsize=(12,4))
    ax = fig.add_subplot(111)
    labels = ['mfrd','mfol','yfrd','yfol']
    edges = split_tri_counts(counts_path)[far]
    friends = []
    lines = []
    for d in edges:
        sets = [set(d[k]) for k in labels]
        frs = len(reduce(set.union,sets))
        if not frs: continue
        friends.append(frs)
        lines.append(.5*sum(len(s) for s in sets)/frs-1)
    ax.plot(friends,lines,'o',
            color='k',
            alpha=.01,
            markersize=13,
            )
    ax.set_xlabel('mutual people')
    ax.set_ylabel('edges')
    ax.set_ylim(0,1)
    ax.set_xlim(0,50)
    fig.savefig('../www/thick_%s.png'%('far' if far else 'near'))


def graph_split_counts(counts_path="tri_counts"):
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    bins=[int(2**(x-1)) for x in xrange(10)]
    for edges,style in zip(split_tri_counts(counts_path),['solid','dotted']):
        pairs = itertools.product(('mfrd','mfol'),('yfrd','yfol'))
        #pairs = (('mfrd','yfrd'),('mfol','yfol'))
        for pair,color in zip(pairs,'rgbk'):
            counts = [len(set(e[pair[0]])&set(e[pair[1]])) for e in edges]
            ax.hist(counts,
                histtype='step',
                label=','.join(pair),
                color=color,
                bins=range(50),
                linestyle=style,
                cumulative=True,
                )
    ax.set_xlabel('count of users in both sets')
    ax.set_ylabel('users')
    ax.set_ylim(0,1000)
    ax.legend()
    fig.savefig('../www/split_pairs.png')


def split_count_friends(counts_path="tri_counts"):
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    styles = ['solid','dotted','dashed']
    for edges,style in zip(split_tri_counts(counts_path),styles):
        keys = ('lmfrd','lmfol','lyfrd','lyfol')
        for key,color in zip(keys,'rgbk'):
            counts = [e[key] for e in edges]
            ax.hist(counts,
                histtype='step',
                label=key,
                color=color,
                bins=range(1000),
                linestyle=style,
                cumulative=True,
                )
            print [style,key,numpy.median(counts)]
    ax.set_xlabel('frds+fols')
    ax.set_ylabel('users')
    #ax.set_ylim(0,4000)
    ax.legend()
    fig.savefig('../www/split_fol_hist.png')


def graph_tri_count_label(counts_path="tri_counts",label='mfan',me='mfol',you='yfol'):
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    last_bin=0
    dist_ratio = [
        (d['dist'], 1.0*len(set(d[me]).intersection(d[you]))/d['all'])
        for d in mainstream_edges(_read_json(counts_path))
        if d['all']
    ]
    for bin in [.2,.4,.6,.8,1]:
        dists = [d for d,r in dist_ratio if last_bin<r<=bin]
        ax.hist(dists,
            bins=numpy.arange(0,100,.2),
            histtype='step',
            #normed=True,
            label="%f-%f"%(last_bin,bin),
            cumulative=True,
            )
        last_bin=bin
    ax.legend()
    #ax.set_ylim(0,1)
    ax.set_ylim(0,2000)
    ax.set_xlabel('length of edge in miles')
    ax.set_ylabel('users')
    fig.savefig('../www/tri_'+label+'_ratio_main.png')


def bar_graph(counts_path="tri_counts"):
    spots = numpy.arange(5)
    width = .35
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    for e,color in zip(split_tri_counts(counts_path),"rb"):
        try:
            r = sum(1 for d in e if d['rfriend'])
        except:
            import pdb;
            pdb.post_mortem()
        strength = [0]*4
        for d in e:
            sets = [set(d[k]) for k in ['mfrd','mfol','yfrd','yfol']]
            uids = reduce(set.union,sets)
            if uids:
                strengths = (sum(u in s for s in sets) for u in uids)
                strength[max(strengths)-1]+=1
            else:
                strength[0]+=1
        ax.bar(spots,[r]+strength,width,color=color)
        spots = spots+width
    ax.set_xticks(spots-width)
    ax.set_xticklabels(('R','0','2','3','4'))
    ax.set_ylabel('users')
    fig.savefig('../www/bar.png')


def count_friends(users_path="hou_tri_users"):
    users, uids = _tri_users_dict_set(users_path)
    logging.info("looking at %d users",len(users))
    fols = []
    frds = []
    for u in users:
        obj = Edges.get_id(int(u))
        if obj._id==None:
            continue
        frds.append(len(uids.intersection(obj.friends)))
        fols.append(len(uids.intersection(obj.followers)))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(fols,bins=range(100),histtype='step')
    ax.hist(frds,bins=range(100),histtype='step')
    fig.savefig('../www/fol_hist.png')


def find_tris(fake=False):
    Edges.database = connect("houtx_edges")
    uids = set(int(d['_id']) for d in _triangle_set())
    logging.info("looking at %d users",len(uids))
    edges = {}
    for uid in uids:
        try:
            obj = Edges.get_id(str(uid))
        except ResourceNotFound:
            edges[uid]=set()
        edges[uid] = uids.intersection(
                (int(f) for f in obj.friends),
                (int(f) for f in obj.followers),
                )
    operation = set.difference if fake else set.intersection
    for me in uids:
        for friend in edges[me]:
            amigos = operation(edges[me],edges[friend])
            for amigo in amigos:
                if friend>amigo:
                    print " ".join(str(id) for id in (me, friend, amigo))


def _coord_params(p1, p2):
    "estimate the distance between two points in miles"
    mlat = 69.1
    mlng = mlat*math.cos((p1['lat']+p2['lat'])*math.pi/360)
    return (mlat*(p1['lat']-p2['lat']), mlng*(p1['lng']-p2['lng']))


def _coord_angle(p, p1, p2):
    "find the angle between rays from p to p1 and p2, return None if p in (p1,p2)"
    vs = [_coord_params(p,x) for x in (p1,p2)]
    mags = [numpy.linalg.norm(v) for v in vs]
    if any(m==0 for m in mags):
        return math.pi
    cos = numpy.dot(*vs)/mags[0]/mags[1]
    return math.acos(min(cos,1))*180/math.pi


def _coord_in_miles(p1, p2):
    return math.hypot(*_coord_params(p1,p2))


def tri_users():
    for user in _triangle_set():
        small = user['gnp']
        small['id'] = user['_id']
        print json.dumps(small)


def _read_json(path=None):
    file = open(path) if path else sys.stdin
    return (simplejson.loads(l) for l in file)


def tri_legs(out_path='tri_hou.png',tris_path="tris",users_path="tri_users.json"):
    if users_path=="couch":
        User.database = connect("houtx_user")
        users = ModelCache(User)
        def user_find(id):
            return users[id].geonames_place.to_d()
    else:
        users = dict((d['id'],d) for d in _read_json(users_path))
        def user_find(id):
            return users[id]

    logging.info("read users")
    xs,ys,zs = [],[],[]
    angs = []
    obtuse=0
    distinct=0
    for line in open(tris_path):
        if random.random()>44324/341318.0: continue
        if len(ys)%10000 ==0:
            logging.info("read %d",len(ys))
        tri = [user_find(id) for id in line.split()]
        dy,dx = sorted([_coord_in_miles(tri[0],tri[x]) for x in (1,2)])
        #dy,dx = [_coord_in_miles(tri[0],tri[x]) for x in (1,2)]
        angle = _coord_angle(*tri)
        if dx<70 and dy<70:
            if angle>90:
                dx,dy=dy,dx
                obtuse+=1
            if angle:
                angs.append(angle)
                distinct+=1
            ys.append(dy)
            xs.append(dx)
            zs.append(_coord_in_miles(tri[1],tri[2]))
    logging.info("read %d triangles",len(ys))
    print obtuse,distinct
    fig = plt.figure(figsize=(18,18))
    ax = fig.add_subplot(223)
    ax.plot(xs,ys,',',
            color='k',
            alpha=.1,
            markersize=10,
            )
    for arr,spot in ((xs,221),(ys,224)):
        ah = fig.add_subplot(spot)
        ah.hist(arr,140,cumulative=True)
        ah.set_xlim(0,70)
        ah.set_ylim(0,50000)
    ah = fig.add_subplot(222)
    ah.hist(angs,90)
    ah.set_xlim(0,180)
    ah.set_ylim(0,1000)
    fig.savefig('../www/'+out_path)

def rfriends():
    Model.database = connect('houtx_edges')
    edges = Model.database.paged_view('_all_docs',include_docs=True)
    clowns = 0
    nobodys = 0
    for row in edges:
        d = row['doc']
        if len(d['frs'])>2000 and len(d['fols'])>2000:
            clowns+=1
            continue
        rfs = set(d['frs']).intersection(d['fols'])
        if not rfs:
            nobodys+=1
            continue
        print " ".join([d['_id']]+list(rfs))
    logging.info("clowns: %d",clowns)
    logging.info("nobodys: %d",nobodys)
