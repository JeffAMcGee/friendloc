from itertools import chain, izip
import datetime
import tempfile
import subprocess

import networkx as nx
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


@gob.mapper(all_items=True,slurp={'connected_ids':set})
def connected_ats(tweets,connected_ids):
    # pick out the time of mentions between connected users
    for tweet in tweets:
        uid = tweet['user']['id']
        if uid not in connected_ids:
            continue
        for mention in tweet['entities']['user_mentions']:
            if mention['id'] in connected_ids:
                yield uid,mention['id'],tweet['created_at']


@gob.mapper(all_items=True)
def daily_ats(connected_ats):
    for frm, to, time in connected_ats:
        format="%a %b %d %H:%M:%S +0000 %Y"
        utc = datetime.datetime.strptime(time,format)
        day = (utc-datetime.timedelta(hours=9)).strftime('%Y-%m-%d')
        yield day,(frm,to)


@gob.mapper(all_items=True)
def user_locs(users):
    for user in users:
        if 'ploc' in user:
            yield user['_id'],user['ploc']


@gob.mapper(all_items=True,slurp={'user_locs_cat':dict})
def near_edges(daily_ats,user_locs_cat):
    edges = {
        (frm,to)
        for frm,to in daily_ats
        if frm in user_locs_cat and to in user_locs_cat
    }
    def _as_array(is_to,is_lat):
        return np.array([user_locs_cat[edge[is_to]][is_lat] for edge in edges])
    flngs = _as_array(0,0)
    flats = _as_array(0,1)
    tlngs = _as_array(1,0)
    tlats = _as_array(1,1)
    dists = utils.np_haversine(flngs,tlngs,flats,tlats)
    for dist,edge in izip(dists,edges):
        if dist<500:
            yield edge[0],edge[1],25/(25+dist)


@gob.mapper(all_items=True)
def mcl_edges(edges):
    with tempfile.NamedTemporaryFile() as abc:
        for edge in edges:
            print>>abc, "%d %d %d"%edge
        abc.flush()
        out = subprocess.check_output(
            ['mcl',abc.name,'--abc','-o','-'],
        )
    for line in out.split('\n'):
        uids = line.split('\t')
        if len(uids)>1:
            yield [int(uid) for uid in uids]

@gob.mapper(all_items=True)
def weak_edges(edges):
    ats = nx.DiGraph()
    for frm,to,dist in edges:
        if dist>.5:
            ats.add_edge(frm,to)
    scc = nx.weakly_connected_components(ats)
    return (c for c in scc if len(c)>1)



