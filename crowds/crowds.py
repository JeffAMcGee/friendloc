from itertools import chain, izip
import datetime
import collections

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
        if dist<50:
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
    for subg in nx.weakly_connected_component_subgraphs(nx.k_core(g,2)):
        if len(subg)>2:
            yield json_graph.adjacency_data(subg)


@gob.mapper(all_items=True)
def find_crowds(weak_comps):
    for crowd,weak_comp in enumerate(weak_comps):
        g = json_graph.adjacency_graph(weak_comp)
        dendo = community.generate_dendogram(nx.Graph(g))

        if len(dendo)>=2:
            partition = community.partition_at_level(dendo, 1 )
            crowds = collections.defaultdict(list)
            for uid,subcrowd in partition.iteritems():
                crowds[subcrowd].append(uid)
            for subcrowd,uids in sorted(crowds.iteritems()):
                subg = nx.subgraph(g,uids)
                subg.graph['crowd'] = (crowd,subcrowd)
                yield json_graph.adjacency_data(subg)
        else:
            g.graph['crowd'] = (crowd,0)
            yield json_graph.adjacency_data(g)


@gob.mapper(all_items=True)
def cluster_crowds(crowds):
    gs = [json_graph.adjacency_graph(g) for g in crowds]
    spots = []
    for index,g in enumerate(gs):
        locs = [ data['loc'] for uid,data in g.nodes_iter(data=True) ]
        lng,lat = np.mean(locs,axis=0)
        g.graph['loc'] = lng,lat
        g.graph['id'] = index
        spots.append((lng,lat))

    sc = cluster.DBSCAN(.2,1)
    clust_ids = sc.fit_predict(np.array(spots))
    clusts = collections.defaultdict(list)
    for clust_id,g in zip(clust_ids,gs):
        clusts[clust_id].append(g)

    clumps = [v for k,v in clusts.iteritems() if k!=-1]
    # crowds that don't fit anywhere go into cluster -1
    extras = [[crowd] for crowd in clusts[-1]]
    clustered = clumps + extras

    for index,clust in enumerate(clustered):
        user_locs = [
                data['loc']
                for g in clust
                for uid,data in g.nodes_iter(data=True)
            ]
        lng,lat = np.median(user_locs,axis=0)
        yield dict(
                id=index,
                loc=(lng,lat),
                size=len(user_locs),
                crowds=[json_graph.adjacency_data(g) for g in clust],
        )


@gob.mapper(all_items=True)
def save_topic(crowd_clusters):
    clusts = []
    for clust_d in crowd_clusters:
        cids = [dict(crowd['graph'])['id'] for crowd in clust_d['crowds']]
        clust = models.Cluster(
                    _id = clust_d['id'],
                    loc = clust_d['loc'],
                    size = clust_d['size'],
                    cids = cids,
        )
        clusts.append(clust)
    topic = models.Topic( _id='conv', clusters = clusts )
    topic.save()


@gob.mapper(all_items=True)
def save_crowds(clusters):
    for clust in clusters:
        for crowd_ in clust['crowds']:
            crowd = json_graph.adjacency_graph(crowd_)
            c = models.Crowd(
                    _id = crowd.graph['id'],
                    loc = crowd.graph['loc'],
                    edges = crowd.edges(),
                    uids = crowd.nodes(),
                )
            c.save()


def _user_crowds(cluster_crowds):
    "create a mapping from user ids to cluster ids given cluster_crowds"
    crowds = [
        json_graph.adjacency_graph(c)
        for clust in cluster_crowds
        for c in clust['crowds']
    ]
    return {
        user:crowd.graph['id']
        for crowd in crowds
        for user in crowd
    }


@gob.mapper(all_items=True,slurp={'cluster_crowds':_user_crowds})
def save_users(user_ds,cluster_crowds):
    # FIXME: allow renaming slurped stuff
    user_crowds = cluster_crowds
    for user_d in user_ds:
        if user_d['id'] not in user_crowds:
            continue
        user = models.User(user_d)
        user.crowd_id = user_crowds[user._id]
        user.merge()


@gob.mapper(all_items=True,slurp={'cluster_crowds':_user_crowds})
def save_tweets(tweets,cluster_crowds):
    # FIXME: allow renaming slurped stuff
    user_crowds = cluster_crowds
    for tweet in tweets:
        uid = tweet['user']['id']
        if uid not in user_crowds or tweet.get('retweeted_status'):
            continue
        cid = user_crowds[uid]

        neighbor_mentions = any(
            user_crowds.get(mention['id'])==cid and mention['id']!=uid
            for mention in tweet['entities']['user_mentions']
        )

        if neighbor_mentions:
            t = models.Tweet(tweet)
            t.crowd_id = cid
            t.save()

