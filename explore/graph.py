import itertools
import random
import logging
import math
import simplejson
import re
import bisect
from collections import defaultdict, namedtuple
from datetime import datetime as dt
from datetime import timedelta
from operator import itemgetter

import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import Bbox
from matplotlib.patches import Patch
import numpy

from settings import settings
import base.twitter as twitter
import base.gisgraphy as gisgraphy
from base.models import *
from maroon import ModelCache
from explore.fixgis import gisgraphy_mdist
from base.utils import *


def graph_hist(data,path,kind="sum",figsize=(12,8),legend_loc=None,normed=False,
        sample=None, histtype='step', marker='-', logline_fn=None,
        label_len=False, auto_ls=False, dist_scale=False, **kwargs):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    if not isinstance(data,dict):
        data = {"":data}

    hargs = {}
    if kind == 'power':
        ax.set_xscale('log')
        ax.set_yscale('log')
        hargs['log']=True
    elif kind == 'linear':
        pass
    elif kind == 'logline':
        ax.set_xscale('log')
        if dist_scale:
            ax.set_yscale('log')
            legend_loc = 3
        elif legend_loc is None:
            legend_loc = 9
    elif kind == 'cumulog':
        ax.set_xscale('log')
        hargs['cumulative']=True
        if legend_loc is None:
            legend_loc = 4
    else:
        hargs['cumulative']=True
        if legend_loc is None:
            legend_loc = 4

    for known in ['bins']:
        if known in kwargs:
            hargs[known] = kwargs[known]

    for index,key in enumerate(sorted(data.iterkeys())):
        if isinstance(key,basestring):
            hargs['label'] = key
        else:
            for k,v in zip(['label','color','linestyle','linewidth'],key):
                hargs[k] = v

        if auto_ls:
            hargs['linestyle'] = ('solid','dashed','dashdot','dotted')[index/7]

        row = data[key]
        if sample:
            row = random.sample(row,sample)
        if normed:
            weight = 1.0/len(row)
            hargs['weights'] = [weight]*len(row)
        if label_len:
            hargs['label'] = "%s (%d)"%(hargs['label'],len(row))

        if kind=="logline":
            for k in ['weights','log','bins']:
                if k in hargs:
                    del hargs[k]
            hist,b = numpy.histogram(row,kwargs['bins'])
            if normed:
                hist = hist*weight
            if dist_scale:
                step_size = b[2]/b[1]
                hist = hist/b[1:]*(step_size/(step_size-1))
            if logline_fn:
                logline_fn(ax, row, b, hist)
            ax.plot((b[:-1]+b[1:])/2, hist, marker, **hargs)
        else:
            ax.hist(row, histtype=histtype, **hargs)
    if normed and kind!='logline':
        ax.set_ylim(0,1)
    elif 'ylim' in kwargs:
        ax.set_ylim(0,kwargs['ylim'])
    if 'xlim' in kwargs:
        try:
            ax.set_xlim(*kwargs['xlim'])
        except TypeError:
            ax.set_xlim(0,kwargs['xlim'])
    if len(data)>1:
        ax.legend(loc=legend_loc)
    ax.set_xlabel(kwargs.get('xlabel'))
    ax.set_ylabel(kwargs.get('ylabel'))
    fig.savefig('../www/'+path,bbox_inches='tight')

def graph_results(path="results"):
    linestyle = defaultdict(lambda: 'solid')
    linestyle['median'] = 'dotted'
    linestyle['omniscient'] = 'dotted'
    linestyle['geocoding'] = 'dotted'
    data = defaultdict(list)
    for block in read_json(path):
        for k,v in block.iteritems():
            data[(k,None,linestyle[k])].extend(v)
    graph_hist(data,
            "results.png",
            bins=dist_bins(120),
            xlim=(1,30000),
            kind="cumulog",
            xlabel = "error in prediction in miles",
            ylabel = "number of users",
            )


def filter_rtt(path=None):
    format = "%a %b %d %H:%M:%S +0000 %Y"
    for tweet in read_json(path):
        if tweet.get('in_reply_to_status_id'):
            ca = time.mktime(dt.strptime(t[2],format).timetuple())
            print "%d\t%d\t%s"%(
                tweet['id'],
                tweet['uid'],
                tweet['in_reply_to_status_id'],
                tweet['created_at']
                )

def graph_rtt(path=None):
    reply = namedtuple("reply",['id','uid','rtt','ca'])
    file = open(path) if path else sys.stdin
    tweets = [reply([int(f) for f in t])
        for t in (line.strip().split('\t') for line in file)]
    time_id = {}
    for t in tweets:
        time_id[t.ca] = max(t.id, time_id.get(t.ca,0))
    cas = sorted(time_id.iterkeys())
    ids = [time_id[ca] for ca in cas]
    deltas = [
        t.ca - cas[bisect.bisect_left(ids,t.rtt)]
        for t in tweets[len(tweets)/2:]
        if t.rtt>ids[0]]
    seen = set()
    graph_hist(deltas,
            "reply_time_sum",
            bins=xrange(0,3600*12,60),
            xlabel="seconds between tweet and reply",
            ylabel="count of tweets in a minute",
        )


def compare_mdist():
    X, Y = [], []
    for user in read_json('data/edges_json'):
        amigo = user.get('rfrd')
        if amigo and amigo['mdist']<100:
            dist = coord_in_miles(user['mloc'],amigo)
            r, theta = dist/amigo['mdist'], random.uniform(0,math.pi/2)
            X.append(r*math.cos(theta))
            Y.append(r*math.sin(theta))

    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    ax.plot(X, Y, '+',
            color='k',
            alpha='.1',
            markersize=4,
            )
    ax.set_xlim(0,2)
    ax.set_ylim(0,2)
    fig.savefig('../www/mdist_plot.png')
    

def fake_dist():
    data = numpy.append(numpy.random.rand(10**5),1/numpy.random.rand(10**5))
    graph_hist(data,
            "fake_dist.png",
            #bins=numpy.arange(0,10,.1),
            bins = numpy.insert(3**numpy.linspace(-5,5,101),0,0),
            dist_scale = True,
            normed=True,
            #kind="linear",
            kind="logline",
            )

def _plot_dist_model(ax, row, *ignored):
    inner = 1.0*sum(1 for r in row if r<1)/len(row)
    ax.plot([.001,1],[inner,inner],'-')

    bins = 10**numpy.linspace(0,1,10)
    hist,bins = numpy.histogram(row,bins)
    step_size = bins[2]/bins[1]
    centers = numpy.sqrt(bins[1:]*bins[:-1])
    #scale for distance and the width of the bucket
    line = hist/bins[1:] * (step_size/(step_size-1)/len(row))
    a,b = numpy.polyfit(numpy.log10(centers),numpy.log10(line),1)
    
    #data = [(10**b)*(x**a) for x in bins]
    data = (10**b) * (bins**a)
    ax.plot(bins, data, '-', color='r')
  


def _plot_dist_model(ax, row, *ignored):
    inner = 1.0*sum(1 for r in row if r<1)/len(row)
    ax.plot([.001,1],[inner,inner],'-',color='k',alpha=.5)

    bins = 10**numpy.linspace(0,1,10)
    hist,bins = numpy.histogram(row,bins)
    step_size = bins[2]/bins[1]
    centers = numpy.sqrt(bins[1:]*bins[:-1])
    #scale for distance and the width of the bucket
    line = hist/bins[1:] * (step_size/(step_size-1)/len(row))
    a,b = numpy.polyfit(numpy.log10(centers),numpy.log10(line),1)
    
    #data = [(10**b)*(x**a) for x in bins]
    data = (10**b) * (bins**a)
    ax.plot(bins, data, '-', color='k',alpha=.5)


def compare_edge_types(cmd=""):
    #set flags
    if cmd=="cuml":
        cuml, prot, mdist = True, False, False
        kwargs = dict(
            bins = dist_bins(120),
            )
    elif cmd=="prot":
        cuml, prot, mdist = True, True, False
        kwargs = dict(
            bins = dist_bins(),
            )
    elif cmd=="mdist":
        cuml, prot, mdist = False, False, True
        kwargs = dict(
            bins = numpy.insert(10**numpy.linspace(-1.975,2.975,100),0,0),
            dist_scale = True,
            logline_fn = _plot_dist_model,
            )
    else:
        cuml, prot, mdist = False, False, False
        kwargs = dict(
            bins = dist_bins(120),
            )

    labels = ('just followers','just friends','recip friends','just mentioned')
    keys = ['rfrd']#('jfol','jfrd','rfrd','jat')
    colors = "gbrc"
    data = defaultdict(list)
    edges = list(read_json('data/edges_json'))
    for user in edges:
        for key,label,color in zip(keys,labels,colors):
            #protected amigo
            pam = user.get('p'+key)
            if prot and pam and pam['mdist']<1000:
                dist = coord_in_miles(user['mloc'],pam)
                data[(label,color,'solid')].append(dist)
 
            #public amigo
            amigo = user.get(key)
            if amigo and amigo['mdist']<1000:
                dist = coord_in_miles(user['mloc'],amigo)
                if mdist:
                    dist = dist/amigo['mdist']
                if not cuml and not mdist:
                    dist+=1
                data[(label,color,'dashed' if prot else 'solid')].append(dist)
    if not cuml and not mdist:
        data[('random rfrd','k')] = 1 + shuffled_dists(edges)
    graph_hist(data,
            "edge_types_%s.png"%cmd,
            xlim=(.01,1000) if mdist else (1,30000),
            normed=True,
            label_len=True,
            kind="cumulog" if cuml else "logline",
            xlabel = "distance between edges in miles",
            ylabel = "number of users",
            **kwargs
            )


def shuffled_dists(edges,kind="rfrd"):
    good = [e for e in edges if kind in e and e[kind]['mdist']<1000]
    dists = (
        coord_in_miles(me['mloc'],you[kind])
        for me,you in zip(good, random.sample(good, len(good)))
        )
    return numpy.fromiter(dists, float, len(good))


def gen_rand_dists():
    users = list(User.find_connected())
    collection = User.database.User
    dests = dict(
        (user['_id'],user['gnp'])
        for user in collection.find({'gnp':{'$exists':1}},fields=['gnp']))
    keys = ('just_followers','just_friends','rfriends') 
    for user in users:
        for key in keys:
            amigo_id = getattr(user,key)[0]
            if dests[amigo_id]['mdist']<1000:
                other = random.choice(users)
                print coord_in_miles(other.median_loc,dests[amigo_id])


def find_urls():
    #for Krishna
    start = dt(2011,2,22,0)
    tweets = Tweet.find(Tweet.created_at.range(start,start+timedelta(hours=2)))
    for t in tweets:
        print t.to_d(long_names=True,dateformat="%a %b %d %H:%M:%S +0000 %Y")

def tweets_over_time():
    tweets = Tweet.find(
            (Tweet.text//r'twitpic\.com/\w+')&
            Tweet.created_at.range(dt(2011,2,19),dt(2011,3,1)),
            fields=['ca'])
    days = [tweet.created_at.hour/24.0+tweet.created_at.day for tweet in tweets]
    graph_hist(
            days,
            "twitpic_hr_lim",
            kind="linear",
            xlabel = "March 2011, UTC",
            ylabel = "tweets with twitpic per hour",
            bins=numpy.arange(19,29,1/24.0),
            xlim=(21,29),
            )


def simplify_tris():
    #reads files generated by print_tri_counts
    for edge_type in ['rfrd','fol','frd','jat']:
        path = 'geo_tri_%s'%edge_type
        out = open('geo_%s_simp'%edge_type,'w')
        counts = []
        for edge in read_json(path):
            if edge['mdist']>100:
                continue
            mfrd = set(edge['mfrd'])
            mfol = set(edge['mfol'])
            yfrd = set(edge['yfrd'])
            yfol = set(edge['yfol'])
            print >>out,json.dumps(dict(
                uid=edge['uid'],
                aid=edge['aid'],
                dist=edge['dist'],
                star=len((mfrd&yfrd) - (mfol|yfol)),
                norm=len(mfol|yfol),
                ))


def dist_bins(per_decade=10):
    return numpy.insert(10**numpy.linspace(0,5,1+5*per_decade),0,0)


def all_ratio_subplot(ax, edges, key, ated):
    CUTOFF=settings.local_max_dist
    BUCKETS=settings.fol_count_buckets
    for kind,color in [['folc','r'],['frdc','b']]:
        dists = defaultdict(list)
        for edge in edges:
            amigo = edge.get(key)
            if amigo and amigo['ated']==ated and amigo['mdist']<1000:
                dist = coord_in_miles(edge['mloc'],amigo)
                bits = min(BUCKETS-1, int(math.log((amigo[kind] or 1),4)))
                dists[bits].append(dist)
        users = 0
        for i in xrange(BUCKETS):
            row = dists[i]
            if not row: continue
            height = 1.0*sum(1 for d in row if d<CUTOFF)/len(row)
            ax.bar(users,height,len(row),color=color,edgecolor=color,alpha=.3)
            users+=len(row)
    ax.set_xlim(0,users)
    ax.set_ylim(0,.8)


def gr_ratio_all():
    edges = list(read_json('edges_json'))
    print "read edges"
    
    conv_labels = [
            "Public I talk to",
            "Public I ignore",
            "Protected I talk to",
            "Protected I ignore"]
    edge_labels = ('just followers','recip friends','just friends','just mentioned')
    edge_types = ('jfol','rfrd','jfrd','jat')

    fig = plt.figure(figsize=(24,12))
    for row, edge_type, edge_label in zip(range(4), edge_types, edge_labels):
        for col,conv_label in enumerate(conv_labels):
            ated = (col%2==0)
            if not ated and row==3:
                continue # this case is a contradiction
            ax = fig.add_subplot(4,4,1+col+row*4)
            edge_key = 'p'+edge_type if col>=2 else edge_type

            all_ratio_subplot(ax, edges, edge_key, ated)
            if col==0:
                ax.set_ylabel(edge_label)
            if row==3:
                ax.set_xlabel('count of users')
            elif row==0:
                ax.set_title('%s users I %s'%(
                    'protected' if col>=2 else 'public', 
                    'talk to' if ated else 'ignore',
                    ))
            ax.tick_params(labelsize="small")
            print 'graphed %d,%d'%(row,col)
    ax = fig.add_subplot(4,4,16,frame_on=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(
        [Patch(color=c,alpha=.3) for c in "rb"],
        ("follower count","friend count"),
        4)
    fig.savefig('../www/local_all.pdf',bbox_inches='tight')


def prep_nonloc(x):
    cutoff = 25
    return (x[cutoff:])


def _kinds_for_et(edge_type):
    #if edge_type in ['frd','jat']:
        return [['star','r'],['norm','b']]
    #else:
    #    return [['sum','g']]

def local_ratio_subplot(ax, dists, rand_dists, rand_nonloc, edge_type, iat, youat):
    for kind,color in _kinds_for_et(edge_type):
        users = 0
        rows = [dists.get((kind,i,iat,youat),[]) for i in xrange(16)]
        size = sum(len(r) for r in rows)
        for row in rows:
            if not row: continue
            hist,b = numpy.histogram(row,dist_bins())
            fit = numpy.polyfit(rand_nonloc, prep_nonloc(hist), 1)
            height = 1-fit[0]*len(rand_dists)/len(row)
            ax.bar(users,height,len(row),color=color,edgecolor=color,alpha=.3)
            users+=len(row)
    ax.set_xlim(0,users)


def gr_local_ratio():
    #reads files generated by simplify_tris, gen_rand_dists, and find_geo_ats
    ats = dict((d['uid'],set(d['ats'])) for d in read_json('geo_ats.json'))
    print "read ats"
    #ats = ModelCache(Tweets,fields=['ats'])
    rand_dists = [float(s.strip()) for s in open('rand_all')]
    print "read rand"
    rand_hist,b = numpy.histogram(rand_dists,dist_bins())
    rand_nonloc = prep_nonloc(rand_hist)

    conv_labels = [
            "I talk to",
            "who converse",
            "who talk to me",
            "who ignore each other"]
    edge_labels = dict(rfrd="rfriends",frd="friends",fol="followers",jat="mentioned")

    fig = plt.figure(figsize=(24,12))
    for row,edge_type in enumerate(['fol','rfrd','frd','jat']):
        dists = defaultdict(list)
        users = defaultdict(int)
        for d in read_json('geo_%s_simp'%edge_type):
            iat = d['aid'] in ats.get(d['uid'],())
            youat = d['uid'] in ats.get(d['aid'],())
            for kind,ignore in _kinds_for_et(edge_type):
                count = d['star']+d['norm'] if kind=='sum' else d[kind]
                bits = (1+int(math.log(count,2))) if count else 0
                dists[(kind,bits,iat,youat)].append(d['dist'])
            users[(iat,youat)]+=1
        total_users = sum(users.values())
        for col,conv_label in enumerate(conv_labels):
            if row==3 and col>1:
                continue #we can't get data for these boxes
            ax = fig.add_subplot(4,4,1+col+row*4)
            
            iat, youat = col<2,1<=col<3
            width = users[(iat,youat)]**2/total_users
            ax.bar(0,.05,width,.95,color='.5')

            local_ratio_subplot(ax,dists,rand_dists,rand_nonloc,edge_type,iat, youat)
            if col==0:
                ax.set_ylabel('local users')
            if row==3:
                ax.set_xlabel('count of users')
            ax.set_ylim(0,1)
            ax.tick_params(labelsize="x-small")
            ax.set_title('%s %s'%(edge_labels[edge_type],conv_label))
            print 'grahped %d,%d'%(row,col)
    fig.savefig('../www/local_ratio_med.png')


def gr_locals(edge_type='rfrd'):
    if edge_type=='all':
        for et in ['frd','fol','rfrd']:
            gr_locals(et)
        return
    counts = list(read_json('geo_%s_simp'%edge_type))
    bins=dist_bins()
    data = {}
    rand_dists = [float(s.strip()) for s in open('rand_all')]
    rand_hist,b = numpy.histogram(rand_dists,bins)
    rand_sum = prep_nonloc(rand_hist)

    for field,color in zip(("star","norm"),'rb'):
        #sort is stable - make it not so stable
        random.shuffle(counts)
        counts.sort(key=itemgetter(field))
        split = len(counts)/4

        steps = zip(
            [counts[i*split:(i+1)*split] for i in xrange(4)],
            ('solid','solid','solid','dotted'),
            (4,2,1,1),
            )
        for part,style,lw in steps:
            row = [1+d['dist'] for d in part]
            hist,b = numpy.histogram(row,bins)
            fit = numpy.polyfit(rand_sum, prep_nonloc(hist), 1)

            label = "%s %d-%d %.2f"%(field, part[0][field], part[-1][field],1-12*fit[0])
            key = (label, color, style, lw)
            data[key]= hist-rand_hist*(fit[0])
    fig = plt.figure(figsize=(18,12))
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    for key in sorted(data.iterkeys()):
        hargs = dict(zip(['label','color','linestyle','linewidth'],key))
        #data[key] = data[key]/sum(data[key])
        #data[key] = numpy.cumsum(data[key])
        ax.plot((bins[:-1]*bins[1:])**.5, data[key], '-', **hargs)
    ax.set_xlim(1,30000)
    ax.set_ylim(0,800)
    ax.legend(loc=9)
    ax.set_xlabel("1+distance between edges in miles")
    ax.set_ylabel("number of users")
    fig.savefig("../www/md_geo_local_"+edge_type)


def gr_split_types(edge_type='rfrd'):
    counts = list(read_json('geo_%s_simp'%edge_type))
    data = {}
    for field,color in zip(("star","norm"),'rb'):
        #sort is stable - make it not so stable
        random.shuffle(counts)
        counts.sort(key=itemgetter(field))
        split = len(counts)/4

        steps = zip(
            [counts[i*split:(i+1)*split] for i in xrange(4)],
            ('solid','solid','solid','dotted'),
            (4,2,1,1),
            )
        for part,style,lw in steps:
            label = "%s %d-%d"%(field, part[0][field], part[-1][field])
            key = (label, color, style, lw)
            data[key] = [1+d['dist'] for d in part]
    graph_hist(data,
            "geo_tri_"+edge_type+"_sp",
            #bins=numpy.insert(10**numpy.linspace(0,5,51),0,0),
            bins=dist_bins(),
            kind="logline",
            xlim=(1,30000),
            xlabel = "1+distance between edges in miles",
            ylabel = "number of users",
            )


def gr_tri_degree(key="mfrd",top=200,right=800):
    top,right=int(top),int(right)
    data = numpy.zeros((top+1,right+1))
    for d in read_json('geo_tri_counts'):
        data[ min(top,len(d[key])), min(right,d['l'+key]) ]+=1
    data = data.clip(0,50)
    #plot it
    fig = plt.figure(figsize=(24,6))
    ax = fig.add_subplot(111)
    ax.imshow(data,
            interpolation='nearest',
            cmap = LinearSegmentedColormap.from_list("gray",["w","k"]),
            )
    ax.set_xlabel("edges")
    ax.set_ylabel("edges in triangle")
    #ax.set_title("Tweets from Houston, TX (11/26/2010-1/14/2010)")
    fig.savefig('../www/tri_deg_%s.png'%key)


def eval_mdist(item=2,kind=5):
    item,kind = int(item),int(kind)
    # this is broken
    gisgraphy_mdist(item,kind)
    users = (User(u) for u in read_json('gnp_gps_46'))
    mdist = gisgraphy.GisgraphyResource()
    dists = []
    mdists = []
    for user in users:
        md = mdist.mdist(user.geonames_place)
        d = coord_in_miles(user.geonames_place.to_d(), user.median_loc)
        dists.append(1+d)
        mdists.append(1+md)
        if 35<md<40:
            print md
    
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    ax.loglog(dists, mdists, '+',
            color='k',
            alpha='.05',
            markersize=10,
            )
    ax.set_xlim(1,30000)
    ax.set_ylim(1,30000)
    ax.set_xlabel("dist")
    ax.set_ylabel("mdist")
    fig.savefig('../www/mdist_%d_%d.png'%(item,kind))
    

def diff_gnp_gps():
    users = (User(u) for u in read_json('gnp_gps_46'))
    mdist = gisgraphy.GisgraphyResource()
    dists = defaultdict(list)
    labels = ["1",'10','100','1000']
    for user in users:
            d = coord_in_miles(user.geonames_place.to_d(),user.median_loc)
            md = mdist.mdist(user.geonames_place)
            bin = len(str(int(md))) if md>=1 else 0
            for key in labels[bin:]:
                dists['PLE<'+key].append(d)
            dists[('all','k','solid',2)].append(d)
            dists[('PLE','.6','dashed',1)].append(md)
    graph_hist(dists,
            "diff_gnp_gps.eps",
            bins = dist_bins(120),
            kind="cumulog",
            normed=True,
            label_len=True,
            xlim=(1,30000),
            xlabel = "location error in miles",
            ylabel = "fraction of users",
            )


def gr_rfr_tris():
    "compare distance from me to you, our rfriend, and my rfriend"
    data = defaultdict(list)
    for quad in read_json('rfrd_tris'):
        for key in ['you','our','my']:
            dist = coord_in_miles(quad['me']['loc'],quad[key]['loc'])
            data[key].append(1+dist)
    graph_hist(data,
            "rfr_tris",
            bins=dist_bins(),
            xlim=(1,30000),
            kind="logline",
            xlabel = "distance between edges in miles",
            ylabel = "number of users",
            )


def mine_ours_img():
    bins = dist_bins(40)
    data = numpy.zeros((len(bins),len(bins)))
    for quad in read_json('rfr_triads'):
        if quad['my']['loc']['mdist']<100  and quad['our']['loc']['mdist']<100:
            spot = [
                bisect.bisect_left(bins, 1+coord_in_miles(quad['me']['loc'],quad[key]['loc']))
                for key in ('our','my')
            ]
            data[tuple(spot)]+=1
    
    data = data-numpy.transpose(data)
    fig = plt.figure(figsize=(24,24))
    ax = fig.add_subplot(111)
    ax.imshow(data,interpolation='nearest')

    ax.set_xlim(0,160)
    ax.set_ylim(0,160)
    ax.set_xlabel("mine")
    ax.set_ylabel("ours")
    ax.set_title("closed vs. open triads")
    fig.savefig('../www/mine_ours.png')
    

def plot_mine_ours():
    data = dict(our=[],my=[])
    for quad in read_json('rfr_triads'):
        if quad['my']['loc']!=quad['our']['loc'] and quad['my']['loc']['mdist']<100  and quad['our']['loc']['mdist']<100:
            for key in data:
                dist = coord_in_miles(quad['me']['loc'],quad[key]['loc'])
                data[key].append(1+dist)
    
    fig = plt.figure(figsize=(24,24))
    ax = fig.add_subplot(111)
    ax.loglog(data['our'],data['my'],'+',
            color='k',
            alpha=.05,
            markersize=10,
            )
    ax.set_xlim(1,30000)
    ax.set_ylim(1,30000)
    ax.set_xlabel("ours")
    ax.set_ylabel("mine")
    ax.set_title("closed vs. open triads")
    fig.savefig('../www/mine_ours.png')
    

def graph_from_net(net):
    edges = [(fol,rfr['_id'])
        for rfr in net['rfrs']
        for fol in rfr['fols']]
    g = nx.DiGraph(edges)
    g.add_nodes_from(
        (r['_id'], dict(lat=r['lat'],lng=r['lng']))
        for r in net['rfrs'])
    return g


def rel_prox():
    data = defaultdict(list)
    colors = dict(zip((4**x for x in xrange(0,7)),'rgbcmyk'))
    for net in read_json('rfr_net_10k'):
        g = graph_from_net(net)
        for fn,tn in itertools.product(g,g):
            if fn>=tn: continue
            edist = coord_in_miles(g.node[fn], g.node[tn])
            bin = 4**int(math.log(1+edist,4))
            dists = sorted(
                coord_in_miles(net['mloc'],g.node[n])
                for n in (tn,fn))
            data["%d_min"%bin,colors[bin],'solid'].append(1+dists[0])
            data["%d_max"%bin,colors[bin],'dashed'].append(1+dists[1])
            data["rfr",'k','dotted',3].append(1+edist)
    graph_hist(data,
            "rel_prox",
            bins=dist_bins(80),
            xlim=(1,30000),
            #kind="logline",
            kind="cumulog",
            label_len=True,
            normed=True,
            xlabel = "distance between edges in miles",
            ylabel = "number of users",
            )

def near_edges_nearby():
    labels = ["0-1",'1-10','10-100','100-1000','1000+']

    data = defaultdict(list)
    for net in read_json('rfr_net_10k'):
        g = graph_from_net(net)
        for fn,tn in itertools.product(g,g):
            if fn==tn: continue
            edist = coord_in_miles(g.node[fn], g.node[tn])
            real = g.has_edge(fn,tn)
            color = 'r' if real else 'b'
            bin = min(4,len(str(int(edist))) if edist>=1 else 0)
            label = '%s %s'%('real' if real else 'fake',labels[bin])
            dist = coord_in_miles(net['mloc'],g.node[fn])
            data[label,color,'solid',1.4**bin].append(1+dist)
    graph_hist(data,
            "near_near",
            bins=dist_bins(80),
            xlim=(1,30000),
            kind="cumulog",
            label_len=True,
            normed=True,
            xlabel = "distance between edges in miles",
            ylabel = "number of users",
            )


def diff_deg():
    data = defaultdict(list)
    for net in read_json('rfr_net_10k'):
        g = graph_from_net(net)
        if g.size()<3: continue
        indegs = sorted(g.out_degree_iter(),key=itemgetter(1))
        for index,key in [(0,"not connected"),(-1,"in-deg connected")]:
            loc = g.node[indegs[index][0]]
            dist = coord_in_miles(net['mloc'],loc)
            data[key].append(1+dist)
    graph_hist(data,
            "out_deg",
            bins=dist_bins(),
            xlim=(1,30000),
            ylim=4000,
            kind="logline",
            xlabel = "distance between edges in miles",
            ylabel = "number of users",
            )


def draw_net_map():
    size=10
    counter = 0
    fig = plt.figure(figsize=(size*4,size*2))
    for net in read_json('rfr_net'):
        counter+=1
        g = graph_from_net(net)
        if not g.size(): continue
        ax = fig.add_subplot(size,size,counter,frame_on=False)
        ax.bar(net['mloc'][0]-.5,1,1,net['mloc'][1]-.5,edgecolor='b')
        pos = dict((r['_id'],(r['lng'],r['lat'])) for r in net['rfrs'])
        nx.draw_networkx_nodes(g,
                ax=ax,
                pos=pos,
                alpha=.1,
                node_size=50,
                edgecolor='r',
                node_shape='d',
                )
        nx.draw_networkx_edges(g,
                ax=ax,
                pos=pos,
                alpha=.2,
                width=1,
                )
        ax.set_xlim(-126,-66)
        ax.set_ylim(24,50)
        ax.set_xticks([])
        ax.set_yticks([])
        if counter ==size*size:
            break
    fig.savefig('../www/net_map.png')


###########################################################
# methods from localcrawl - use at your own risk!
#

def plot_tweets():
    #usage: peek.py print_locs| peek.py plot_tweets
    locs = read_gis_locs()
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
    ax.plot(noisy(lngs,.005),noisy(lats,.005),',',
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
    for t in read_json(path):
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


def graph_edges(users_path="hou_tri_users", ats_path="hou_ats"):
    users, uids = tri_users_dict_set(users_path)
    ats,ated = parse_ats(ats_path)
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
            dists = [coord_in_miles(users[u],users[other]) for other in others]
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


def graph_split_counts(counts_path="geo_tri_counts"):
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
    #ax.set_ylim(0,1000)
    ax.legend()
    fig.savefig('../www/geo_split_pairs.png')


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


def graph_tri_count_label(label='mfan',me='mfol',you='yfol'):
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    last_bin=0
    dist_ratio = [
        (d['dist'], 1.0*len(set(d[me]).intersection(d[you]))/d['all'])
        for d in mainstream_edges(read_json('geo_tri_counts'))
        if d['all']
    ]
    for bin in [.2,.4,.6,.8,1]:
        dists = [d for d,r in dist_ratio if last_bin<r<=bin]
        ax.hist(dists,
            bins=range(2000),#numpy.arange(0,100,.2),
            histtype='step',
            #normed=True,
            label="%f-%f"%(last_bin,bin),
            cumulative=True,
            )
        last_bin=bin
    ax.legend()
    #ax.set_ylim(0,1)
    ax.set_ylim(0,3000)
    ax.set_xlabel('length of edge in miles')
    ax.set_ylabel('users')
    fig.savefig('../www/geo_'+label+'_ratio.png')


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
    users, uids = tri_users_dict_set(users_path)
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
    uids = set(int(d['_id']) for d in triangle_set())
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



def tri_users():
    for user in triangle_set():
        small = user['gnp']
        small['id'] = user['_id']
        print json.dumps(small)



def tri_legs(out_path='tri_hou.png',tris_path="tris",users_path="tri_users.json"):
    if users_path=="couch":
        User.database = connect("houtx_user")
        users = ModelCache(User)
        def user_find(id):
            return users[id].geonames_place.to_d()
    else:
        users = dict((d['id'],d) for d in read_json(users_path))
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
        dy,dx = sorted([coord_in_miles(tri[0],tri[x]) for x in (1,2)])
        #dy,dx = [coord_in_miles(tri[0],tri[x]) for x in (1,2)]
        angle = coord_angle(*tri)
        if dx<70 and dy<70:
            if angle>90:
                dx,dy=dy,dx
                obtuse+=1
            if angle:
                angs.append(angle)
                distinct+=1
            ys.append(dy)
            xs.append(dx)
            zs.append(coord_in_miles(tri[1],tri[2]))
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
