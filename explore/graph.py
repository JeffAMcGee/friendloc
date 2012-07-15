OUTPUT_TYPE = None # 'png', 'pdf', or None

import random
import logging
import math
import bisect
import contextlib
from collections import defaultdict, namedtuple
from datetime import datetime as dt
from datetime import timedelta

import networkx as nx
import matplotlib

# this needs to happen before pyplot is imported - it cannot be changed
if OUTPUT_TYPE:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy

from settings import settings
import base.gisgraphy as gisgraphy
#from base.models import *
#from base.utils import *
from base import gob


@contextlib.contextmanager
def axes(path='', figsize=(12,8), legend_loc=4,
         xlabel=None, ylabel=None, xlim=None, ylim=None, ):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    yield ax
    if xlim is not None:
        try:
            ax.set_xlim(*xlim)
        except TypeError:
            ax.set_xlim(0,xlim)
    if ylim is not None:
        try:
            ax.set_ylim(*ylim)
        except TypeError:
            ax.set_ylim(0,ylim)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if legend_loc:
        ax.legend(loc=legend_loc)
    if OUTPUT_TYPE:
        fig.savefig('../www/'+path+'.'+OUTPUT_TYPE,bbox_inches='tight')


def linhist(ax, row, bins, dist_scale=False, window=None, normed=False,
            marker='-', **hargs):
    "works like ax.hist, but without jagged edges"
    hist,b = numpy.histogram(row,bins)
    step_size = b[2]/b[1]
    hist = hist*(1.0*step_size/(step_size-1))
    if window is not None:
        hist = numpy.convolve(hist,window,mode='same')/sum(window)
    if normed:
        hist = hist * (1.0/len(row))
    if dist_scale:
        hist = hist/b[1:]
        ax.set_yscale('log')
    ax.set_xscale('log')
    ax.plot((b[:-1]+b[1:])/2, hist, marker, **hargs)

def ugly_graph_hist(data,path,kind="sum",figsize=(12,8),legend_loc=None,normed=False,
        sample=None, histtype='step', marker='-', logline_fn=None,
        label_len=False, auto_ls=False, dist_scale=False, ax=None,
        window=None, ordered_label=False, **kwargs):
    # DEPRECATED - TOO COMPLEX!
    if ax:
        fig = None
    else:
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
                if v is not None:
                    hargs[k] = v
        if ordered_label:
            hargs['label'] = hargs['label'][2:]

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
            step_size = b[2]/b[1]
            hist = hist*(1.0*step_size/(step_size-1))
            if window is not None:
                hist = numpy.convolve(hist,window,mode='same')/sum(window)
            if normed:
                hist = hist*weight
            if dist_scale:
                hist = hist/b[1:]
            if logline_fn:
                logline_fn(ax, row, b, hist)
            ax.plot((b[:-1]+b[1:])/2, hist, marker, **hargs)
        else:
            ax.hist(row, histtype=histtype, **hargs)
    if normed and kind!='logline':
        ax.set_ylim(0,1)
    elif 'ylim' in kwargs:
        try:
            ax.set_ylim(*kwargs['ylim'])
        except TypeError:
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
    if fig is not None:
        fig.savefig('../www/'+path,bbox_inches='tight')


def graph_results(path="results"):
    linestyle = defaultdict(lambda: 'solid')
    linestyle['Median'] = 'dotted'
    linestyle['Omniscient *'] = 'dotted'
    linestyle['Mode'] = 'dotted'
    data = defaultdict(list)
    for block in read_json(path):
        for k,v in block.iteritems():
            k = k.replace('lul','contacts')
            data[(k,None,linestyle[k])].extend(v)
    for k,v in data.iteritems():
        print k[0], 1.0*sum(1 for d in v if d<25)/len(v)
    ugly_graph_hist(data,
            "top_results.pdf",
            bins=dist_bins(120),
            xlim=(1,30000),
            kind="cumulog",
            normed=True,
            ordered_label=True,
            xlabel = "error in prediction in miles",
            ylabel = "fraction of users",
            )


def diff_mode_fl():
    mode=[]
    fl=[]
    for block in read_json('results'):
        mode.extend(block['Mode'])
        fl.extend(block['FriendlyLocation'])
    mode = numpy.array(mode)+1
    fl = numpy.array(fl)+1

    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    ax.loglog(mode, fl, '+',
            color='k',
            alpha='.05',
            markersize=5,
            )
    ax.set_xlim(1,30000)
    ax.set_ylim(1,30000)
    ax.set_xlabel("mode")
    ax.set_ylabel("fl")
    fig.savefig('../www/mode_fl.png')

 
def filter_rtt(path=None):
    format = "%a %b %d %H:%M:%S +0000 %Y"
    for tweet in read_json(path):
        if tweet.get('in_reply_to_status_id'):
            #ca = time.mktime(dt.strptime(t[2],format).timetuple())
            print "%d\t%d\t%s"%(
                tweet['id'],
                tweet['uid'],
                tweet['in_reply_to_status_id'],
                tweet['created_at']
                )


def graph_rtt(path=None):
    #FIXME: This is for crowdy, not FriendlyLocation!
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
    ugly_graph_hist(deltas,
            "reply_time_sum",
            bins=xrange(0,3600*12,60),
            xlabel="seconds between tweet and reply",
            ylabel="count of tweets in a minute",
        )


def _plot_dist_model(ax, row, *ignored):
    inner = 1.0*sum(1 for r in row if r<1)/len(row)
    ax.plot([.001,1000],[inner,inner],'-',color='k',alpha=.2)

    bins = 10**numpy.linspace(0,1,11)
    hist,bins = numpy.histogram(row,bins)
    step_size = bins[2]/bins[1]
    centers = numpy.sqrt(bins[1:]*bins[:-1])
    #scale for distance and the width of the bucket
    line = hist/bins[1:] * (step_size/(step_size-1)/len(row))
    a,b = numpy.polyfit(numpy.log(centers),numpy.log(line),1)
    
    #data = [(10**b)*(x**a) for x in bins]
    X = 10**numpy.linspace(0,2,21)
    Y = (math.e**b) * (X**a)
    ax.plot(X, Y, '-', color='k',alpha=.2)

CONTACT_GROUPS = dict(
    jfol = dict(label='just followers',color='g'),
    jfrd = dict(label='just friends',color='b'),
    rfrd = dict(label='recip friends',color='r'),
    jat = dict(label='just mentioned',color='c'),
)

@gob.mapper(all_items=True)
def graph_edge_types_cuml(edge_dists):
    data = defaultdict(list)

    for key,dists in edge_dists:
        conf = CONTACT_GROUPS[key[0]]
        data[(conf['label'],conf['color'],'solid')].extend(dists)

    for k,v in data.iteritems():
        print k,sum(1.0 for x in v if x<25)/len(v)
    ugly_graph_hist(data,
            "edge_types_cuml.png",
            xlim= (1,30000),
            normed=True,
            label_len=True,
            kind="cumulog",
            ylabel = "fraction of users",
            xlabel = "distance to contact in miles",
            bins = dist_bins(120),
            )

@gob.mapper(all_items=True)
def graph_edge_types_prot(edge_dists):
    data = defaultdict(list)

    for key,dists in edge_dists:
        conf = CONTACT_GROUPS[key[0]]
        fill = 'solid' if key[2] else 'dotted'
        data[(conf['label'],conf['color'],fill)].extend(dists)

    ugly_graph_hist(data,
            "edge_types_prot.png",
            xlim = (1,30000),
            normed=True,
            label_len=True,
            kind="cumulog",
            ylabel = "fraction of users",
            xlabel = "distance to contact in miles",
            bins = dist_bins(80),
            )

@gob.mapper(all_items=True)
def graph_edge_types_norm(edge_dists):
    data = defaultdict(list)
    for key,dists in edge_dists:
        conf = CONTACT_GROUPS[key[0]]
        data[(conf['label'],conf['color'],'solid')].extend(dists)
    for key,dists in data.iteritems():
        data[key] = [d+1 for d in dists]

    ugly_graph_hist(data,
            "edge_types_norm.png",
            xlim = (1,30000),
            normed=True,
            label_len=True,
            kind="logline",
            ylabel = "fraction of users",
            xlabel = "distance to contact in miles",
            bins = dist_bins(40),
            ylim = .6,
            window = numpy.bartlett(7),
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
    #for New Zealand project and Krishna
    start = dt(2011,2,22,0)
    tweets = Tweet.find(Tweet.created_at.range(start,start+timedelta(hours=2)))
    for t in tweets:
        print t.to_d(long_names=True,dateformat="%a %b %d %H:%M:%S +0000 %Y")


def tweets_over_time():
    #for the New Zealand project
    tweets = Tweet.find(
            (Tweet.text//r'twitpic\.com/\w+')&
            Tweet.created_at.range(dt(2011,2,19),dt(2011,3,1)),
            fields=['ca'])
    days = [tweet.created_at.hour/24.0+tweet.created_at.day for tweet in tweets]
    ugly_graph_hist(
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
    ats = dict((d['uid'],set(d['ats'])) for d in read_json('geo_ats.json'))
    logging.info("read ats")
    com_types = {
        (False,False):"We ignore",
        (True,False):"I talk",
        (False,True):"You talk",
        (True,True):"We talk",
        }
    for edge_type in ['rfrd','fol','frd','jat']:
        path = 'geo_tri_%s'%edge_type
        out = open('geo_%s_simp'%edge_type,'w')
        for edge in read_json(path):
            if edge['mdist']>1000:
                continue
            iat = edge['aid'] in ats.get(edge['uid'],())
            youat = edge['uid'] in ats.get(edge['aid'],())

            mfrd = set(edge['mfrd'])
            mfol = set(edge['mfol'])
            yfrd = set(edge['yfrd'])
            yfol = set(edge['yfol'])

            star = mfrd&yfrd
            path = mfrd&yfol
            fan  = mfol&yfol
            loop = mfol&yfrd
            print >>out,json.dumps(dict(
                uid=edge['uid'],
                aid=edge['aid'],
                dist=edge['dist'],
                norm=len(mfol|yfol),
                star=len(star - fan),
                fan=len(fan - star),
                path=len(path - loop),
                loop=len(loop - path),
                com_type = com_types[(iat,youat)],
                ))
        logging.info("wrote %s",edge_type)


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


def com_types():
    fig = plt.figure(figsize=(24,12))
    titles = dict(fol="Just Follower", rfrd="Reciprical Friend", frd="Just Friend", jat="Just Mentiened")
    for spot,edge_type in enumerate(['rfrd','frd','fol','jat']):
        ax = fig.add_subplot(2,2,1+spot)
        simp = read_json('geo_%s_simp'%edge_type)
        data = defaultdict(list)
        for d in simp:
            data[d['com_type']].append(d['dist'])
        for k,v in data.iteritems():
            print edge_type,k,sum(1.0 for x in v if x<25)/len(v) 
        ugly_graph_hist(data, "", ax=ax,
                legend_loc=2,
                bins=dist_bins(80),
                kind="cumulog",
                xlim=(1,30000),
                normed=True,
                label_len=True,
                xlabel = "distance between edges in miles",
                ylabel = "number of users",
                )
        ax.set_title(titles[edge_type])
    fig.savefig("../www/com_types.pdf",bbox_inches='tight')


def triad_types():
    fig = plt.figure(figsize=(24,12))
    titles = dict(fol="Just Follower", rfrd="Reciprical Friend", frd="Just Friend", jat="Just Mentiened")
    for col,edge_type in enumerate(['rfrd','frd','fol','jat']):
        ax = fig.add_subplot(2,2,1+col)
        counts = list(read_json('geo_%s_simp'%edge_type))
        data = {}
        for field,color in zip(("star","fan","path","loop"),'rbgk'):
            steps = [
                (False,'dotted',.5,"no"),
                (True,'solid',1,"has"), ]
            for part,style,lw,prefix in steps:
                label = "%s %s"%(prefix,field)
                key = (label, color, style, lw)
                data[key] = [
                    d['dist']
                    for d in counts
                    if part==bool(d[field])]
        ugly_graph_hist(data, "", ax=ax,
                bins=dist_bins(80),
                kind="cumulog",
                xlim=(1,30000),
                label_len=True,
                normed=True,
                xlabel = "distance between edges in miles",
                ylabel = "number of users",
                )
        ax.set_title(titles[edge_type])
    fig.savefig("../www/triad_types.pdf",bbox_inches='tight')


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
    ugly_graph_hist(dists,
            "diff_gnp_gps.eps",
            bins = dist_bins(120),
            kind="cumulog",
            normed=True,
            label_len=True,
            xlim=(1,30000),
            xlabel = "location error in miles",
            ylabel = "fraction of users",
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
