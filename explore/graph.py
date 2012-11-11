OUTPUT_TYPE = None # 'png', 'pdf', or None
OUTPUT_TYPE = 'png'#, 'pdf', or None

import random
import contextlib
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


class VectFit(object):
    def __init__(self,env):
        self.env = env

    @gob.mapper(all_items=True)
    def graph_vect_fit(self, vect_fit, in_paths):
        if in_paths[0][-1] != '0':
            return
        ratios = (ratio for cutoff,ratio in self.env.load('vect_ratios.0'))
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
        backstrom="Backstrom",
        last="Random Contact",
        #median="Median Contact",
        nearest="Nearest Predicted Contact",
        friendloc_basic="FriendlyLocation Basic",
        friendloc_full="FriendlyLocation Full",
        omni="Omniscient",
    )
    _gr_preds(preds,labels,'gr_basic.png')


@gob.mapper(all_items=True)
def gr_parts(preds):
    labels = dict(
        backstrom="Backstrom Baseline",
        friendloc_plain="FriendlyLocation Basic",
        friendloc_cut="FriendlyLocation with Cutoff",
        friendloc_tz="FriendlyLocation with UTC offset",
        friendloc_field="FriendlyLocation with Location field",
        omni="Omniscient Baseline",
    )
    _gr_preds(preds,labels,'gr_parts.png')


def _gr_preds(preds,labels,path):
    data = {labels[key]:val for key,val in preds}
    for key,vals in data.iteritems():
        print key,sum(1 for v in vals if v<25)

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
    jfol = dict(label='just followers',color='g'),
    jfrd = dict(label='just friends',color='b'),
    rfrd = dict(label='recip friends',color='r'),
    jat = dict(label='just mentioned',color='c'),
)


@gob.mapper(all_items=True)
def graph_edge_types_cuml(edge_dists):
    data = defaultdict(list)

    for key,dists in edge_dists:
        if key[0]=='rand':
            continue
        conf = CONTACT_GROUPS[key[0]]
        data[(conf['label'],conf['color'],'solid')].extend(dists)

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
        if key[0]=='rand':
            continue
        conf = CONTACT_GROUPS[key[0]]
        fill = 'solid' if key[-1] else 'dotted'
        data[(conf['label'],conf['color'],fill)].extend(dists)

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
        if key[0]=='rand':
            continue
        conf = CONTACT_GROUPS[key[0]]
        data[(conf['label'],conf['color'],'solid')].extend(dists)
    for key,dists in data.iteritems():
        data[key] = [d+1 for d in dists]

    ugly_graph_hist(data,
            "edge_types_norm.pdf",
            xlim = (1,15000),
            normed=True,
            label_len=True,
            kind="logline",
            ylabel = "fraction of users",
            xlabel = "distance to contact in miles",
            bins = dist_bins(40),
            ylim = .6,
            )


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
            data[(conf['label'],conf['color'],'solid')].append(dist)

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
def graph_locals(rfr_dists):
    def _bucket(ratio):
        if ratio is None:
            return None
        elif 0<=ratio<.25:
            return "0.0<=ratio<.25"
        elif ratio<.5:
            return "0.25<=ratio<.5"
        elif ratio<.75:
            return "0.5<=ratio<.75"
        assert ratio<=1
        return "0.75<=ratio<=1"

    data=dict(
        lofrd = defaultdict(list),
        lofol = defaultdict(list),
        cheap = defaultdict(list),
        dirt = defaultdict(list),
    )

    for amigo in rfr_dists:
        for color,key in (('r','dirt'),('b','lofol'),('g','cheap')):
            label = _bucket(amigo[key])
            if label is None:
                data[key][('No leafs','k','dotted')].append(amigo['dist'])
            else:
                data[key][label].append(amigo['dist'])

    titles = dict(
                cheap="Contacts with Local Contacts",
                dirt="10 Contacts with Local Contacts",
                #lofrd="Contacts with Local Friends",
                lofol="Contacts with Local Followers")
    fig = plt.figure(figsize=(18,6))
    for spot,key in enumerate(('lofol','cheap','dirt')):
        ax = fig.add_subplot(1,3,1+spot,title=titles[key])

        for subkey,dists in data[key].iteritems():
            print key, subkey, 1.0*sum(1 for d in dists if d<25)/len(dists)

        ugly_graph_hist(data[key],
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

    fig.savefig("../flt/figures/local_ratio.pdf",bbox_inches='tight')



@gob.mapper(all_items=True)
def graph_com_types(edge_dists):
    data = defaultdict(lambda: defaultdict(list))

    for key,dists in edge_dists:
        if key[0]=='rand':
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
                xlim=(1,15000),
                label_len=True,
                normed=True,
                xlabel = "distance between edges in miles",
                ylabel = "number of users",
                )
        ax.set_title(titles[edge_type])
    fig.savefig("../www/triad_types.pdf",bbox_inches='tight')


@gob.mapper(all_items=True)
def graph_mloc_mdist(mloc_mdists):
    dists = defaultdict(list)
    labels = ["1",'10','100','1000']
    for mloc,mdist in mloc_mdists:
            bin = len(str(int(mdist))) if mdist>=1 else 0
            for key in labels[bin:]:
                dists['PLE<'+key].append(mloc)
            dists[('all','k','solid',2)].append(mloc)
            dists[('PLE','.6','dashed',1)].append(mdist)
    for key,vals in dists.iteritems():
        print key,sum(1 for v in vals if v<1000)

    ugly_graph_hist(dists,
            "mloc_mdist.pdf",
            bins = dist_bins(120),
            kind="cumulog",
            normed=True,
            label_len=True,
            xlim=(1,15000),
            xlabel = "location error in miles",
            ylabel = "fraction of users",
            )


@gob.mapper(all_items=True)
def near_triads(rfr_triads):
    labels = ["",'0-10','10-100','100-1000','1000+']
    data = defaultdict(list)

    for quad in rfr_triads:
        for key,color in (('my','r'),('our','b')):
            edist = coord_in_miles(quad[key]['loc'],quad['you']['loc'])
            bin = min(4,len(str(int(edist))))
            label = '%s %s'%(key,labels[bin])
            dist = coord_in_miles(quad[key]['loc'],quad['me']['loc'])
            # 1.6**(bin-1) is the line width calculation
            data[label,color,'solid',1.6**(bin-1)].append(dist)
    ugly_graph_hist(data,
            "near_triads.pdf",
            bins=dist_bins(120),
            xlim=(1,15000),
            label_len=True,
            kind="cumulog",
            normed=True,
            xlabel = "distance between edges in miles",
            ylabel = "number of users",
            )
