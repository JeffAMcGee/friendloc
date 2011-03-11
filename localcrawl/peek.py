#!/usr/bin/env python
# This is a tool for investigating the collected data.  It is designed to
# be %run in ipython.  If you import it from another module, you're doing
# something wrong.

import itertools
import time
import os
import sys
import random
import logging
import getopt
import math
from collections import defaultdict
from datetime import datetime as dt
from operator import itemgetter
import cjson

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import Bbox
import numpy

from couchdbkit import ResourceNotFound

from settings import settings
import twitter
from models import *
from maroon import ModelCache
from scoredict import Scores, BUCKETS, log_score


def all_users():
    return Model.database.paged_view('_all_docs',include_docs=True,endkey='_')


def place_tweets(start, end):
    return Model.database.paged_view('tweet/plc',
            include_docs=True,
            startkey=None,
            endkey=None,
            startkey_docid=start,
            endkey_docid=end,
            )


def count_users(key):
    counts = defaultdict(int)
    for u in all_users():
        counts[u['doc'].get(key,None)]+=1
    for k in sorted(counts.keys()):
        print "%r\t%d"%(k,counts[k])


def count_locations(path='counts'):
    counts = defaultdict(int)
    for u in all_users():
        if u['doc'].get('prob',0)==1:
            loc = u['doc'].get('loc',"")
            norm = " ".join(re.split('[^0-9a-z]+', loc.lower())).strip()
            counts[norm]+=1
    f = open(path,'w')
    for k,v in sorted(counts.iteritems(),key=itemgetter(1)):
        print>>f, "%r\t%d"%(k,v)
    f.close()


def count_tweets_in_box(start='T',end='U'):
    counts = defaultdict(int)
    box = settings.local_box
    for row in place_tweets(start,end):
        if 'coord' in row['doc']:
            c = row['doc']['coord']['coordinates']
            if box['lng'][0]<c[0]<box['lng'][1] and box['lat'][0]<c[1]<box['lat'][1]:
                counts['inb']+=1
            else:
                counts['outb']+=1
        else:
            counts['noco']+=1
    print dict(counts)


def print_locs(start='T',end='U'):
    for row in place_tweets(start,end):
        if 'coord' in row['doc']:
            c = row['doc']['coord']['coordinates']
            print '%f\t%f\t%s'%(c[0],c[1],row['doc']['uid'])


def read_locs(path=None):
    for l in open(path or "locs_uid"):
        s = l.split()
        yield float(s[0]),float(s[1]),s[2]
    logging.info("read points")



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


def count_recent():
    min_int_id = 8000000000000000L
    view = Model.database.paged_view('user/and_tweets')
    for k,g in itertools.groupby(view, lambda r:r['key'][0]):
            user_d = g.next()
            if user_d['id'][0] != 'U':
                print "fail %r"%user_d
                continue
            tweets = sum(1 for r in g if int(r['id'])>min_int_id)
            print "%d\t%s"%(tweets,user_d['id'])
         

def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.izip_longest(*args, fillvalue=fillvalue)


def count_sn(path):
    "used to evaluate the results of localcrawl"
    lost =0
    found =0
    sns = (sn.strip() for sn in open(path))
    for group in grouper(100,sns):
        for user in res.user_lookup([], screen_names=group):
            if user._id in Model.database:
                found+=1
                print "found %s - %s"%(user.screen_name,user._id)
            else:
                lost+=1
                print "missed %s - %s"%(user.screen_name,user._id)
    print "lost:%d found:%d"%(lost,found)


def analyze():
    "Find out how the scoring algorithm did."
    scores = Scores()
    scores.read(settings.lookup_out)

    locs = (0,.5,1)
    weights =(.1,.3,.5,.7,.9)
    counts = dict(
        (score, dict(
            (loc, dict(
                (weight,0)
                for weight in weights))
            for loc in locs))
        for score in xrange(BUCKETS))

    for user in User.get_all():
        if user.utc_offset==7200: continue
        state, rfs, ats = scores.split(user._id)

        for weight in weights:
            score = log_score(rfs,ats,weight)
            counts[score][user.local_prob][weight]+=1

    print "non\t\t\t\t\tunk\t\t\t\t\tlocal"
    for score in xrange(BUCKETS):
        for loc in locs:
            for weight in weights:
                print "%d\t"%counts[score][loc][weight],
        print


def krishna_export(start=[2010],end=None):
    "export the tweets for Krishna's crawler"
    view = Model.database.paged_view(
            'tweet/date',
            include_docs=True,
            startkey=start,
            endkey=end
        )
    for k,g in itertools.groupby(view,itemgetter('key')):
        path = os.path.join(*(str(x) for x in k))
        mkdir_p(os.path.dirname(path))
        with open(path,'w') as f:
            for t in (row['doc'] for row in g):
                ts = int(time.mktime(dt(*t['ca']).timetuple()))
                if t['ats']:
                    for at in t['ats']:
                        print>>f,"%d %s %s %s"%(ts,t['_id'],t['uid'],at)
                else:
                    print>>f,"%d %s %s"%(ts,t['_id'],t['uid'])


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

def _tri_users_dict_set(users_path):
    users = dict((int(d['id']),d) for d in _read_json(users_path))
    return users,set(users)

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


def find_ats(users_path="hou_tri_users"):
    users, uids = _tri_users_dict_set(users_path)
    for line in sys.stdin:
        d = json.loads(line)
        if d.get('ats'):
            local = int(d['uid']) in uids
            for at in d['ats']:
                if local or int(at) in uids:
                    print "%s\t%s"%(d['uid'],at)

def _parse_ats(ats_path):
    ats =defaultdict(lambda: defaultdict(int))
    ated =defaultdict(lambda: defaultdict(int))
    for line in open(ats_path):
        uid,at = [int(i) for i in line.strip().split('\t')]
        ats[uid][at]+=1
        ated[at][uid]+=1
    return ats,ated

def print_tri_counts(users_path="hou_tri_users"):
    users, uids = _tri_users_dict_set(users_path)
    edges = ModelCache(Edges)
    data = []
    for uid,user in users.iteritems():
        me = edges[uid]
        if not me : continue
        friends = uids.intersection(me.friends)
        if not friends: continue
        your_id = random.sample(friends,1)[0]
        you = edges[your_id]
        sets = dict(
            mfrd = set(me.friends),
            mfol = set(me.followers),
            yfrd = set(you.friends),
            yfol = set(you.followers),
            )
        all = (sets['mfrd']|sets['mfol'])&(sets['yfrd']|sets['yfol'])
        d = dict(
            dist = _coord_in_miles(user,users[your_id]),
            all = len(all),
            rfriend = 1 if your_id in sets['mfol'] else 0 
        )
        for k,v in sets.iteritems():
            d['l'+k]= len(v)
            d[k] = list(all&v)
        data.append(d)
    data.sort(key=itemgetter('dist'))
    for d in data:
        print json.dumps(d)

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


def edge_dist(users_path="hou_tri_users"):
    users, uids = _tri_users_dict_set(users_path)
    logging.info("looking at %d users",len(users))
    for u in users:
        obj = Edges.get_id(int(u))
        if obj._id==None:
            continue


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
    return (69.1*(p1['lat']-p2['lat']), 60.4*(p1['lng']-p2['lng']))


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


def _read_json(path):
    for l in open(path):
        yield simplejson.loads(l)


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
