from itertools import chain, izip
import datetime
import collections
import math

import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
from sklearn import cluster

from base import gob, utils, models
import community


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


def _filter_users_from_tweets(tweets,filt):
    seen = set()
    for tweet in tweets:
        uid = tweet['user']['id']
        if uid in seen or not filt(uid):
            continue
        seen.add(uid)
        yield models.User.mod_id(uid),tweet['user']


@gob.mapper(all_items=True,slurp={'connected_ids':set})
def connected_users(tweets,connected_ids):
    return _filter_users_from_tweets(tweets, lambda uid: uid in connected_ids)


@gob.mapper(all_items=True,slurp={'connected_ids':set})
def disconnected_users(tweets,connected_ids):
    return _filter_users_from_tweets(tweets, lambda uid: uid not in connected_ids)


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
        if dist<25:
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
def weak_comps(edges,user_locs):
    """convert near_edges into networkx format, keep weakly-connected users"""
    g = nx.DiGraph()
    for edge in edges:
        ne = NearEdge(*edge)
        if 813286 in (ne.frm,ne.to):
            # FIXME: hardcoding this is all kinds of ugly
            # 813286 is @BarackObama, we skip him because he breaks clustering
            # in D.C.
            continue
        conv = (ne.day,ne.at,ne.rt)
        if g.has_edge(ne.frm,ne.to):
            g[ne.frm][ne.to]['conv'].append(conv)
        else:
            g.add_edge(ne.frm, ne.to, dist=ne.dist, conv=[conv])
    for frm,to,data in g.edges(data=True):
        if not any(at for day,at,rt in data['conv']):
            g.remove_edge(frm,to)
    for node in g.nodes_iter():
        g.node[node]['loc'] = user_locs[node]
    for subg in nx.weakly_connected_component_subgraphs(g):
        if len(subg)>2:
            yield json_graph.adjacency_data(subg)


@gob.mapper(all_items=True)
def find_crowds(weak_comps):
    crowds = []
    for crowd,weak_comp in enumerate(weak_comps):
        g = json_graph.adjacency_graph(weak_comp)
        dendo = community.generate_dendogram(nx.Graph(g))

        if len(dendo)>=2:
            partition = community.partition_at_level(dendo, 1 )
            crowd_ids = collections.defaultdict(list)
            for uid,subcrowd in partition.iteritems():
                crowd_ids[subcrowd].append(uid)
            for subcrowd,uids in sorted(crowd_ids.iteritems()):
                subg = nx.subgraph(g,uids)
                crowds.append(subg)
        else:
            crowds.append(g)

    spots = collections.defaultdict(list)
    for index,g in enumerate(crowds):
        locs = [ data['loc'] for uid,data in g.nodes_iter(data=True) ]
        lng,lat = np.mean(locs,axis=0)
        g.graph['loc'] = lng,lat
        g.graph['id'] = index
        spots[int(lng),int(lat)].append(g)

    for lng_lat,graphs in spots.iteritems():
        graphs.sort(key=len,reverse=True)
        for index,g in enumerate(graphs):
            g.graph['zoom'] = int(math.floor(1+math.log(index,4))) if index else 1

    return (json_graph.adjacency_data(g) for g in crowds)


@gob.mapper(all_items=True)
def save_crowds(crowds):
    for crowd_ in crowds:
        crowd = json_graph.adjacency_graph(crowd_)
        c = models.Crowd(
                _id = crowd.graph['id'],
                loc = crowd.graph['loc'],
                zoom = crowd.graph['zoom'],
                edges = crowd.edges(),
                uids = crowd.nodes(),
            )
        c.save()


def _user_crowds(crowds):
    "create a mapping from user ids to cluster ids given cluster_crowds"
    crowds = (
        json_graph.adjacency_graph(c)
        for c in crowds
    )
    return {
        user:crowd.graph['id']
        for crowd in crowds
        for user in crowd
    }


@gob.mapper(all_items=True,slurp={'find_crowds':_user_crowds})
def save_users(user_ds,find_crowds):
    # FIXME: allow renaming slurped stuff
    user_crowds = find_crowds
    for user_d in user_ds:
        if user_d['id'] not in user_crowds:
            continue
        user = models.User(user_d)
        user.crowd_id = user_crowds[user._id]
        user.merge()


@gob.mapper(all_items=True,slurp={'find_crowds':_user_crowds})
def save_tweets(tweets,find_crowds):
    # FIXME: allow renaming slurped stuff
    user_crowds = find_crowds
    for tweet in tweets:
        uid = tweet['user']['id']
        if uid not in user_crowds or tweet.get('retweeted_status'):
            continue
        cid = user_crowds[uid]

        t = models.Tweet(tweet)
        t.crowd_id = cid
        t.user_sn = tweet['user']['screen_name']
        t.profile_image_url = tweet['user']['profile_image_url']
        t.ents = tweet['entities']
        t.save()

