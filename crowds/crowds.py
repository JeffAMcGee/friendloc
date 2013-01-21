from itertools import chain, izip
import datetime
import tempfile
import subprocess
import collections

import networkx as nx
from networkx.readwrite import json_graph
import numpy as np

from base import gob, utils, models


@gob.mapper(all_items=True)
def connected_ids(tweets):
    ats = nx.DiGraph()
    for tweet in tweets:
        uid = tweet['user']['id']
        for mention in tweet['entities']['user_mentions']:
            if mention['id']!=uid:
                ats.add_edge(uid,mention['id'])
    scc = nx.strongly_connected_components(ats)
    return chain.from_iterable( g for g in scc if len(g)>2 )


@gob.mapper(all_items=True,slurp={'connected_ids':set})
def connected_users(tweets,connected_ids):
    seen = set()
    for tweet in tweets:
        uid = tweet['user']['id']
        if uid not in connected_ids or uid in seen:
            continue
        seen.add(uid)
        yield models.User.mod_id(uid),tweet['user']


@gob.mapper(all_items=True,slurp={'connected_ids':set})
def disconnected_users(tweets,connected_ids):
    # FIXME: this only differs from connected_users by removing not from the if
    seen = set()
    for tweet in tweets:
        uid = tweet['user']['id']
        if uid in connected_ids or uid in seen:
            continue
        seen.add(uid)
        yield models.User.mod_id(uid),tweet['user']


@gob.mapper(all_items=True)
def user_locs(predicted_users,cheap_users):
    """Combine connected users with users who gave us a location"""
    for users in (predicted_users,cheap_users):
        for user in users:
            if 'ploc' in user:
                yield user['_id'],user['ploc']


def _date_from_stamp(time):
    format="%a %b %d %H:%M:%S +0000 %Y"
    utc = datetime.datetime.strptime(time,format)
    return (utc-datetime.timedelta(hours=9)).strftime('%Y-%m-%d')


def _key_set(kvs):
    return {k for k,v in kvs}


@gob.mapper(all_items=True,slurp={'user_locs':_key_set})
def daily_ats(tweets,user_locs):
    """split up the mentions and retweets based on the day"""
    for tweet in tweets:
        uid = tweet['user']['id']
        if uid not in user_locs:
            continue
        day = _date_from_stamp(tweet['created_at'])
        rts = tweet.get('retweeted_status')
        if rts:
            if rts['user']['id'] in user_locs:
                yield day,('rt',uid,rts['user']['id'])
        else:
            for mention in tweet['entities']['user_mentions']:
                if mention['id'] in user_locs and mention['id']!=uid:
                    yield day,('at',uid,mention['id'])


NearEdge = collections.namedtuple('NearEdge','frm to dist day at rt')

@gob.mapper(all_items=True,slurp={'user_locs':dict})
def near_edges(daily_ats, user_locs, in_paths):
    # FIXME: I'm really starting to dislike in_paths
    day_name = in_paths[0].split('.')[-1]
    dt = datetime.datetime.strptime(day_name,"%Y-%m-%d")
    day = dt.date() - datetime.date(2012,8,1)

    edges = collections.defaultdict(lambda: collections.defaultdict(int))
    for kind,frm,to in daily_ats:
        if frm in user_locs and to in user_locs:
            edges[frm,to][kind]+=1
    def _as_array(is_to,is_lat):
        return np.array([user_locs[edge[is_to]][is_lat] for edge in edges])
    flngs = _as_array(0,0)
    flats = _as_array(0,1)
    tlngs = _as_array(1,0)
    tlats = _as_array(1,1)
    dists = utils.np_haversine(flngs,tlngs,flats,tlats)

    for dist,(frm,to) in izip(dists,edges):
        if dist<100:
            edge = edges[frm,to]
            yield NearEdge(
                frm,
                to,
                dist,
                day.days,
                edge.get('at',0),
                edge.get('rt',0),
            )


@gob.mapper(all_items=True,slurp={'user_locs':dict})
def weak_comps(edges):
    """convert near_edges into networkx format, keep the weakly-connected nodes"""
    g = nx.DiGraph()
    for edge in edges:
        ne = NearEdge(*edge)
        conv = (ne.day,ne.at,ne.rt)
        if g.has_edge(ne.frm,ne.to):
            g[ne.frm][ne.to]['conv'].append(conv)
        else:
            g.add_edge(ne.frm, ne.to, dist=ne.dist, conv=[conv])
    for node in g.nodes_iter():
        g.node[node]['loc'] = user_locs[node]
    for subg in nx.weakly_connected_component_subgraphs(g):
        if len(subg)>2:
            yield json_graph.adjacency_data(g)


@gob.mapper(all_items=True)
def mcl_edges(edges):
    #BROKEN
    with tempfile.NamedTemporaryFile() as abc:
        for edge in edges:
            print>>abc, "%d %d %d"%edge
        abc.flush()
        out = subprocess.check_output(
            ['mcl',abc.name,'-V','all','--abc','-o','-'],
        )
    for line in out.split('\n'):
        if not line: continue
        uids = line.split('\t')
        if len(uids)>1:
            yield [int(uid) for uid in uids]

@gob.mapper(all_items=True,slurp={'user_locs':dict})
def weak_edges(edges,user_locs):
    #BROKEN
    ats = nx.DiGraph()
    for frm,to,dist in edges:
        if dist>.5:
            ats.add_edge(frm,to)
    wcc = nx.weakly_connected_components(ats)
    return (
            [(id,user_locs[id],ats[id].keys()) for id in clust]
            for clust in wcc
            if len(clust)>2
        )
