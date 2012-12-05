OUTPUT_TYPE = None # 'png', 'pdf', or None
OUTPUT_TYPE = 'pdf'#, 'pdf', or None

import random
import contextlib
import bisect
import math
from collections import defaultdict

import matplotlib

# this needs to happen before pyplot is imported - it cannot be changed
if OUTPUT_TYPE:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy

from explore import peek
#from base.models import *
from base.utils import dist_bins, coord_in_miles
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
        label_len=False, auto_ls=False, ax=None,
        window=None, ordered_label=False, key_order=None, **kwargs):
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
        if legend_loc is None:
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

    if key_order is None:
        key_order = sorted(data.iterkeys())

    for index,key in enumerate(key_order):
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
    if legend_loc:
        ax.legend(loc=legend_loc)
    ax.set_xlabel(kwargs.get('xlabel'))
    ax.set_ylabel(kwargs.get('ylabel'))
    if fig is not None:
        fig.savefig('../www/'+path,bbox_inches='tight')



@gob.mapper(all_items=True)
def graph_vect_fit(vect_fit, in_paths, env):
    if in_paths[0][-1] != '0':
        return
    ratios = (ratio for cutoff,ratio in env.load('vect_ratios.0'))
    fits = (fit for cutoff,fit in vect_fit)

    bins = dist_bins(120)
    miles = numpy.sqrt([bins[x-1]*bins[x] for x in xrange(2,482)])

    with axes('vect_fit',legend_loc=1) as ax:
        ax.set_xlim(1,10000)
        ax.set_ylim(1e-8,2e-3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('distance in miles')
        ax.set_ylabel('probablility of being a contact')

        colors = iter('rgbkm')
        labels = iter([
            'edges predicted in nearest 10%',
            'edges in 60th to 70th percentile',
            'edges in 30th to 40th percentile',
            'edges predicted in most distant 10%',
        ])
        for index,(ratio,fit) in enumerate(zip(ratios, fits)):
            if index%3!=0:
                continue

            color = next(colors)
            label = next(labels)
            window = numpy.bartlett(5)
            smooth_ratio = numpy.convolve(ratio,window,mode='same')/sum(window)
            ax.plot(miles, smooth_ratio, '-', color=color, label=label)
            ax.plot(miles, peek.contact_curve(miles,*fit), '-',
                    linestyle='dashed', color=color)


@gob.mapper(all_items=True)
def gr_basic(preds):
    labels = dict(
        backstrom=("Backstrom Baseline",'r','dotted',2),
        #last="Random Contact",
        #median="Median Contact",
        #mode="Mode of Contacts",
        nearest=("Nearest Predicted Contact",'b','dashed',2),
        #friendloc_plain="FriendlyLocation Basic",
        friendloc_strange=("FriendlyLocation",'k','solid',1),
        #omni="Omniscient",
    )
    _gr_preds(preds,labels,'fl_basic.pdf')

CONTACT_GROUPS = dict(
    jfol = ('just followers','g','dashed',2),
    jfrd = ('just friends','r','dotted',2),
    rfrd = ('recip friends','k','solid',1),
    jat = ('just mentioned','b','dashdot',2),
)


@gob.mapper(all_items=True)
def gr_parts(preds):
    labels = dict(
        backstrom="Backstrom Baseline",
        friendloc_plain="FriendlyLocation Basic",
        friendloc_cut0="FriendlyLocation with Cutoff",
        friendloc_cut="FriendlyLocation with Cutoff +loc +str",
        friendloc_strange="FriendlyLocation with Strange",
        friendloc_tz="FriendlyLocation with UTC offset",
        friendloc_loc="FriendlyLocation with Location field",
        friendloc_full="FriendlyLocation Full",
        omni="Omniscient Baseline",
    )
    _gr_preds(preds,labels,'fl_parts.pdf')

def _aed(ratio,vals):
    return numpy.average(sorted(vals)[:int(ratio*len(vals))])

def _gr_preds(preds,labels,path):
    preds_d = defaultdict(list)
    for key,dists in preds:
        preds_d[key].extend(dists)
    data = {labels[key]:preds_d[key] for key in labels}

    ugly_graph_hist(data,
            path,
            xlim= (1,15000),
            normed=True,
            label_len=True,
            kind="cumulog",
            ylabel = "fraction of users",
            xlabel = "error in prediction (miles)",
            bins = dist_bins(120),
            )


CONTACT_GROUPS = dict(
    jfol = ('just followers','g','dashed',2),
    jfrd = ('just friends','r','dotted',2),
    rfrd = ('recip friends','k','solid',1),
    jat = ('just mentioned','b','dashdot',2),
)


@gob.mapper(all_items=True)
def graph_edge_types_cuml(edge_dists):
    data = defaultdict(list)

    for key,dists in edge_dists:
        if key[0]=='usa':
            continue
        conf = CONTACT_GROUPS[key[0]]
        data[conf].extend(dists)

    for k,v in data.iteritems():
        print k,sum(1.0 for x in v if x<25)/len(v)

    ugly_graph_hist(data,
            "edge_types_cuml.pdf",
            xlim= (1,15000),
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
        if key[0]=='usa':
            continue
        conf = CONTACT_GROUPS[key[0]]
        fill = 'solid' if key[-1] else 'dotted'
        label,color,oldfill,width = conf
        data[(lable,color,fill)].extend(dists)

    for k,v in data.iteritems():
        print k,round(100*sum(1.0 for x in v if x<25)/len(v)),len(v)

    ugly_graph_hist(data,
            "edge_types_prot.pdf",
            xlim = (1,15000),
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
        if key[0]=='usa':
            continue
        data[key[0]].extend(dists)
    smallest = min(len(v) for v in data.itervalues())

    fig = plt.figure(figsize=(12,12))
    for spot,key in enumerate(['rfrd','jfol','jfrd','jat']):
        ax = fig.add_subplot(4,1,1+spot)
        xlabel = "distance between edges in miles" if spot==3 else ''
        label = CONTACT_GROUPS[key][0]
        ugly_graph_hist({label:random.sample(data[key],smallest)},
            'ignored',
            ax=ax,
            bins=dist_bins(40),
            xlim=(1,15000),
            ylim=(0,4000),
            label_len=True,
            legend_loc=2,
            kind="linear",
            xlabel = xlabel,
            ylabel = "number of users",
            )
        ax.set_xscale('log')
        if spot!=3:
            ax.get_xaxis().set_ticklabels([])
    fig.savefig("../www/edge_types_norm.pdf",bbox_inches='tight')


@gob.mapper(all_items=True)
def graph_edge_count(rfr_dists):
    frd_data = defaultdict(list)
    fol_data = defaultdict(list)
    labels = ["",'1-9','10-99','100-999','1000-9999','10000+']
    key_labels = dict(frdc='Friends',folc='Followers')

    for amigo in rfr_dists:
        for color,key,data in (('r','folc',fol_data),('b','frdc',frd_data)):
            bin = min(5,len(str(int(amigo[key]))))
            label = '%s %s'%(labels[bin],key_labels[key])
            # 1.6**(bin-1) is the line width calculation
            data[label,color,'solid',1.6**(bin-2)].append(amigo['dist'])

    fig = plt.figure(figsize=(18,6))
    for spot,data in enumerate((fol_data,frd_data)):
        ax = fig.add_subplot(1,2,1+spot)
        ugly_graph_hist(data,
            'ignored',
            ax=ax,
            bins=dist_bins(120),
            xlim=(1,15000),
            label_len=True,
            kind="cumulog",
            normed=True,
            xlabel = "distance between edges in miles",
            ylabel = "fraction of users",
            )
    fig.savefig("../www/edge_counts.pdf",bbox_inches='tight')


@gob.mapper(all_items=True)
def graph_local_groups(edges):
    data=defaultdict(list)
    for edge in edges:
        for key,conf in CONTACT_GROUPS.iteritems():
            if key not in edge:
                continue
            amigo = edge.get(key)
            if amigo['lofol'] is None or amigo['lofol']<.5:
                continue
            dist = coord_in_miles(edge['mloc'],amigo)
            data[conf].append(dist)

    ugly_graph_hist(data,
            "local_groups.pdf",
            xlim= (1,15000),
            normed=True,
            label_len=True,
            kind="cumulog",
            ylabel = "fraction of users",
            xlabel = "distance to contact in miles",
            bins = dist_bins(120),
            )



@gob.mapper(all_items=True)
def graph_locals_10(rfr_dists):
    labels = [ ".0<=lr<.25", ".25<=lr<.5", ".5<=lr<.75", ".75<=lr<=1" ]
    data = defaultdict(list)

    for amigo in rfr_dists:
        if amigo['dirt'] is None:
            data[('No leafs','k','dotted')].append(amigo['dist'])
        else:
            label = labels[min(3,int(math.floor(amigo['dirt']*4)))]
            data[label].append(amigo['dist'])

    ugly_graph_hist(data,
        'locals_10.pdf',
        bins=dist_bins(120),
        xlim=(1,15000),
        label_len=True,
        kind="cumulog",
        normed=True,
        xlabel = "distance between edges in miles",
        ylabel = "fraction of users",
        )


@gob.mapper(all_items=True)
def graph_locals_cmp(rfr_dists):
    ratio_dists = defaultdict(list)
    for amigo in rfr_dists:
        for key in ('dirt','aint','cheap'):
            if amigo[key] is not None:
                ratio_dists[key].append((amigo[key],amigo['dist']))

    keys = ['dirt','cheap','aint']
    labels = [
        ('10 leafs','b','dashed',2),
        ('20 leafs','r','dotted',2),
        ('100 leafs','k','solid',1),
    ]

    good_dists = {}
    for key,label in zip(keys,labels):
        tups = ratio_dists[key]
        med = numpy.median([ratio for ratio,dist in tups])
        good_dists[label] = [dist for ratio,dist in tups if ratio>=med]

    ugly_graph_hist(good_dists,
        'locals_cmp.pdf',
        bins=dist_bins(120),
        xlim=(1,15000),
        label_len=True,
        kind="cumulog",
        normed=True,
        xlabel = "distance between edges in miles",
        ylabel = "fraction of users",
        key_order = labels,
        )



@gob.mapper(all_items=True)
def graph_com_types(edge_dists):
    data = defaultdict(lambda: defaultdict(list))

    for key,dists in edge_dists:
        if key[0]=='usa':
            continue
        edge_type,i_at,u_at,prot = key
        # ignore protected
        data[edge_type][i_at,u_at].extend(dists)


    titles = dict(
        jfol="Just Follower",
        rfrd="Reciprical Friend",
        jfrd="Just Friend",
        jat="Just Mentiened")
    labels = {
        (False,False):"We ignore",
        (True,False):"I talk",
        (False,True):"You talk",
        (True,True):"We talk",
        }
    fig = plt.figure(figsize=(18,12))

    for edge_type,sub_d in data.iteritems():
        for mentions,dists in sub_d.iteritems():
            print edge_type, mentions, 1.0*sum(1 for d in dists if d<25)/len(dists)

    for spot,edge_type in enumerate(['rfrd','jfrd','jfol','jat']):
        ax = fig.add_subplot(2,2,1+spot)

        # UGLY
        picked = {
            labels[key]:dists
            for key,dists in data[edge_type].iteritems()
        }

        ugly_graph_hist(picked, "", ax=ax,
                legend_loc=2,
                bins=dist_bins(80),
                kind="cumulog",
                xlim=(1,15000),
                normed=True,
                label_len=True,
                xlabel = "distance between edges in miles",
                ylabel = "number of users",
                )
        ax.set_title(titles[edge_type])
    fig.savefig("../www/com_types.pdf",bbox_inches='tight')


CONTACT_GROUPS = dict(
    jfol = ('just followers','b','dashed',2),
    jfrd = ('just friends','b','dotted',2),
    rfrd = ('recip friends','b','solid',1),
    jat = ('just mentioned','b','dashdot',2),
)

@gob.mapper(all_items=True)
def graph_mloc_mdist(mloc_mdists):
    dists = defaultdict(list)
    labels = ["",'0-10','10-100','100-1000','1000+']
    for mloc,mdist in mloc_mdists:
            bin = min(4,len(str(int(mdist))))
            width = .6*1.8**(bin-1)
            dists[labels[bin],'b','solid',width].append(mloc)
            dists[('PLE','k','dashed',1)].append(mdist)
    for key,vals in dists.iteritems():
        print key,sum(1 for v in vals if v<1000)

    ugly_graph_hist(dists,
            "mloc_mdist.pdf",
            bins = dist_bins(120),
            kind="cumulog",
            normed=True,
            label_len=True,
            xlim=(1,30000),
            xlabel = "location error in miles",
            ylabel = "fraction of users",
            )


@gob.mapper(all_items=True)
def near_triads(rfr_triads):
    labels = ["",'0-10','10-100','100-1000','1000+']
    data = defaultdict(list)

    for quad in rfr_triads:
        for key,color,fill in (('my','b','dashed'),('our','r','solid')):
            edist = coord_in_miles(quad[key]['loc'],quad['you']['loc'])
            bin = min(4,len(str(int(edist))))
            label = '%s %s'%(key,labels[bin])
            dist = coord_in_miles(quad[key]['loc'],quad['me']['loc'])
            width = .6*1.8**(bin-1)
            data[label,color,'solid',width].append(dist)
    ugly_graph_hist(data,
            "near_triads.pdf",
            bins=dist_bins(120),
            xlim=(1,30000),
            label_len=True,
            kind="cumulog",
            normed=True,
            xlabel = "distance between edges in miles",
            ylabel = "number of users",
            )


@gob.mapper(all_items=True)
def graph_indep(rfr_indep):
    # This function is ugly.

    bins=dist_bins(120)
    miles = numpy.sqrt([bins[x-1]*bins[x] for x in xrange(2,482)])
    # FIXME: hardcoded values from _fit_stgrs in peek.py
    fit_stgrs = 10**(1.034*(numpy.log10(miles))+6.207)

    data = defaultdict(lambda: numpy.zeros(602))
    for quad in rfr_indep:
        for key,color in (('near','r'),('far','b')):
            d = coord_in_miles(quad['me']['loc'],quad[key]['gnp'])
            data[key][bisect.bisect(bins,d)]+=1

    with axes('indep',legend_loc=1) as ax:
        ax.set_xlim(1,15000)
        #ax.set_ylim(1e-8,2e-3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('distance in miles')
        ax.set_ylabel('probablility of being a contact')

        #window = numpy.bartlett(5)
        #smooth_ratio = numpy.convolve(ratio,window,mode='same')/sum(window)
        for key,vect in data.iteritems():
            ax.plot( miles, vect[2:482]/fit_stgrs, '-', label=key)

