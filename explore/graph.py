import itertools
import random
import logging
import math
import simplejson
import re
import bisect
from collections import defaultdict, namedtuple
from datetime import datetime as dt
from operator import itemgetter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import Bbox
import numpy

from settings import settings
import base.twitter as twitter
import base.gisgraphy as gisgraphy
from base.models import *
from maroon import ModelCache
from base.utils import *



def graph_hist(data,path,kind="sum",figsize=(18,12),legend_loc=None,normed=False,
        sample=None, histtype='step', **kwargs):
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
        if legend_loc is None:
            legend_loc = 9
    elif kind == 'cumulog':
        ax.set_xscale('log')
        hargs['cumulative']=True
        if legend_loc is None:
            legend_loc = 2
    else:
        hargs['cumulative']=True
        if legend_loc is None:
            legend_loc = 4

    for known in ['bins']:
        if known in kwargs:
            hargs[known] = kwargs[known]

    for key in sorted(data.iterkeys()):
        if isinstance(key,basestring):
            hargs['label'] = key
        else:
            for k,v in zip(['label','color','linestyle','linewidth'],key):
                hargs[k] = v
        row = data[key]
        if sample:
            row = random.sample(row,sample)
        if normed:
            hargs['weights'] = [1.0/len(row)]*len(row)

        if kind=="logline":
            for k in ['weights','log','bins']:
                if k in hargs:
                    del hargs[k]
            hist,b = numpy.histogram(row,kwargs['bins'])
            ax.plot((b[:-1]+b[1:])/2, hist, '-', **hargs)
        else:
            ax.hist(row, histtype=histtype, **hargs)
    if normed:
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
    fig.savefig('../www/'+path)


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


def compare_edge_types():
    TwitterModel.database = MongoDB(name='ngeo',slave_okay=True)
    logging.info("read ats")
    users = User.find(
            User.just_friends.exists() &
            User.just_followers.exists() &
            User.rfriends.exists(),
            )
    dests = {}
    for user in User.database.User.find(fields=['gnp']):
        if 'gnp' in user:
            dests[user['_id']] = user['gnp']
    logging.info("read gnp")
    keys = ('just_followers','just_friends','rfriends')
    data = defaultdict(list)
    last_spot = place = [-92.2,37.5] #population center of US, just a sane default
    for user in users:
        for key in keys:
            amigo_id = getattr(user,key)[0]
            place = dests[amigo_id]
            dist = coord_in_miles(user.median_loc,place)
            data[key].append(1+dist)
            rand_dist =coord_in_miles(last_spot,place)
            data['rand-'+key].append(1+rand_dist)
            #print rand_dist
        last_spot = user.median_loc
    for k,v in data.iteritems():
        logging.info("%s %d",k,len(v))
    graph_hist(data,
            "geo_edges_rfr",
            bins=dist_bins(),
            xlim=(1,30000),
            ylim=6000,
            #normed=True,
            kind="logline",
            xlabel = "distance between edges in miles",
            ylabel = "number of users",
            )


def tweets_over_time():
    days =[]
    for tweet in Tweet.find(Tweet.text//r'twitpic\.com/\w+',fields=['ca']):
        ca = tweet.created_at
        days.append(ca.hour/24.0+ca.day)
    graph_hist(
            days,
            "twitpic_hr_lim",
            kind="linear",
            xlabel = "March 2011, UTC",
            ylabel = "tweets with twitpic per hour",
            bins=numpy.arange(4,30,1/24.0),
            )


def simplify_tris(edge_type='rfrd'):
    f = open('geo_%s_simp'%edge_type,'w')
    counts = []
    for edge in read_json():
        mfrd = set(edge['mfrd'])
        mfol = set(edge['mfol'])
        yfrd = set(edge['yfrd'])
        yfol = set(edge['yfol'])
        print >>f,json.dumps(dict(
            uid=edge['uid'],
            aid=edge['aid'],
            dist=edge['dist'],
            star=len((mfrd&yfrd) - (mfol|yfol)),
            norm=len(mfol|yfol),
            ))


def dist_bins():
    return numpy.insert(10**numpy.linspace(0,5,51),0,0)


def prep_nonloc(x):
    cutoff = 25
    return (x[cutoff:])

def local_ratio_subplot(ax, dists, rand_dists, rand_nonloc, edge_type, iat, youat):
    if edge_type=='frd':
        kinds = [['star','r'],['norm','b']]
    else:
        kinds = [['sum','g']]
    for kind,color in kinds:
        users = 0
        for i in xrange(15):
            row = dists.get((kind,i,iat,youat),None)
            if not row: continue
            hist,b = numpy.histogram(row,dist_bins())
            fit = numpy.polyfit(rand_nonloc, prep_nonloc(hist), 1)
            height = 1-fit[0]*len(rand_dists)/len(row)
            ax.bar(users,height,len(row),color=color,edgecolor=color,alpha=.3)
            users+=len(row)
    ax.set_xlim(0,users)


def gr_local_ratio():
    ats = dict((d['uid'],set(d['ats'])) for d in read_json('geo_ats.json'))
    rand_dists = [float(s.strip()) for s in open('rand_all')]
    rand_hist,b = numpy.histogram(rand_dists,dist_bins())
    rand_nonloc = prep_nonloc(rand_hist)

    conv_labels = [
            "I talk to",
            "who converse",
            "who talk to me",
            "who ignore each other"]
    edge_labels = dict(rfrd="rfriends",frd="friends",fol="followers")

    fig = plt.figure(figsize=(15,10))
    for col,edge_type in enumerate(['fol','rfrd','frd']):
        dists = defaultdict(list)
        for d in read_json('geo_%s_simp'%edge_type):
            for kind in ['star','norm','sum']:
                iat = d['aid'] in ats.get(d['uid'],())
                youat = d['uid'] in ats.get(d['aid'],())
                count = d['star']+d['norm'] if kind=='sum' else d[kind]
                bits = (1+int(math.log(count,2))) if count else 0
                dists[(kind,bits,iat,youat)].append(d['dist'])
        for row,conv_label in enumerate(conv_labels):
            ax = fig.add_subplot(3,4,1+row+col*4)
            local_ratio_subplot(ax,dists,
                    rand_dists,rand_nonloc,edge_type,
                    row<2,1<=row<3)
            if row==0:
                ax.set_ylabel('local users')
            if col==2:
                ax.set_xlabel('count of users')
            ax.set_ylim(0,1)
            ax.tick_params(labelsize="x-small")
            ax.set_title('%s %s'%(edge_labels[edge_type],conv_label))
    fig.savefig('../www/local_ratio.png')


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
    #ax.set_ylim(0,1)
    ax.legend(loc=9)
    ax.set_xlabel("1+distance between edges in miles")
    ax.set_ylabel("number of users")
    fig.savefig("../www/geo_local_"+edge_type)


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




def diff_gnp_gps(path=None):
    gis = gisgraphy.GisgraphyResource()
    names = {}
    latlngs = defaultdict(list)
    for t in read_json(path):
        uid = t['user']['id']
        if 'coordinates' not in t: continue
        lng,lat = t['coordinates']['coordinates']
        latlngs[uid].append(dict(lng=lng,lat=lat))
        names[uid] = t['user']['location']
    dists = []
    for uid,spots in latlngs.iteritems():
        if len(spots)<2: continue
        if coord_in_miles(spots[0],spots[1])>30: continue
        gnp = gis.twitter_loc(names[uid])
        if not gnp: continue
        if gnp.population>5000000: continue
        if gnp.feature_code>"ADM1": continue
        dist = coord_in_miles(gnp.to_d(),spots[0])
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
