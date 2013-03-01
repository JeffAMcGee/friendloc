from __future__ import absolute_import

from itertools import chain, izip
import datetime
import collections
import math
import operator

import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
import random

from friendloc.base import gob, utils, models
import crowds.community


@gob.mapper(all_items=True)
def connected_ids(tweets):
    """
    read in tweets and pick out user ids in strongly connected component
    USAGE: zcat /spare/twitwatch/conv/*/*.gz | ./gb.py -s connected_ids
    """
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
    """
    read in tweets and return user dicts for user ids in connected_ids
    USAGE: zcat /spare/twitwatch/conv/*/*.gz | ./gb.py -s connected_users
    """
    # FIXME: rearrange this so we make fewer passes over the data
    return _filter_users_from_tweets(tweets, lambda uid: uid in connected_ids)


@gob.mapper(all_items=True,slurp={'connected_ids':set})
def disconnected_users(tweets,connected_ids):
    """
    read in tweets and return user dicts for user ids not in connected_ids
    USAGE: zcat /spare/twitwatch/conv/*/*.gz | ./gb.py -s disconnected_users
    """
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
    """
    split up the mentions and retweets based on the day
    USAGE: zcat /spare/twitwatch/conv/*/*.gz | ./gb.py -s disconnected_users
    """
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
    """
    keep edges from daily_ats between users who live within 25 miles
    """
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
        conv = (ne.day,ne.at,ne.rt)
        if g.has_edge(ne.frm,ne.to):
            g[ne.frm][ne.to]['conv'].append(conv)
        else:
            g.add_edge(ne.frm, ne.to, dist=ne.dist, conv=[conv])
    # remove edges that are only retweets
    for frm,to,data in g.edges(data=True):
        if not any(at for day,at,rt in data['conv']):
            g.remove_edge(frm,to)
    # remove nodes with a degree greater than 50
    popular = [uid for uid,degree in g.degree_iter() if degree>50]
    g.remove_nodes_from(popular)
    # add locations
    for node in g.nodes_iter():
        g.node[node]['loc'] = user_locs[node]
    # yield weakly connected crowds
    for subg in nx.weakly_connected_component_subgraphs(g):
        if len(subg)>2:
            yield json_graph.adjacency_data(subg)


@gob.mapper(all_items=True)
def find_crowds(weak_comps):
    """
    break up big connected components using crowd detection algorithm, add
    details to crowds
    """
    #FIXME: I think the crowds = [] definition hides the import crowds.community
    crowds = []
    for crowd,weak_comp in enumerate(weak_comps):
        g = json_graph.adjacency_graph(weak_comp)
        dendo = crowds.community.generate_dendogram(nx.Graph(g))

        if len(dendo)>=2:
            partition = crowds.community.partition_at_level(dendo, 1 )
            crowd_ids = collections.defaultdict(list)
            for uid,subcrowd in partition.iteritems():
                crowd_ids[subcrowd].append(uid)
            for subcrowd,uids in sorted(crowd_ids.iteritems()):
                subg = nx.DiGraph(nx.subgraph(g,uids))
                if len(subg)>2:
                    crowds.append(subg)
        else:
            crowds.append(g)

    def _blur(angle):
        return random.triangular(angle-.02,angle+.02)

    spots = collections.defaultdict(list)
    for index,g in enumerate(crowds):
        uid,degree = max(g.degree_iter(),key=operator.itemgetter(1))
        lng,lat = g.node[uid]['loc']
        g.graph['loc'] = _blur(lng),_blur(lat)
        g.graph['id'] = index
        spots[int(lng/2),int(lat/2)].append(g)

    for lng_lat,graphs in spots.iteritems():
        graphs.sort(key=len,reverse=True)
        for index,g in enumerate(graphs):
            g.graph['zoom'] = int(math.floor(1+math.log(index,4))) if index else 0

    return (json_graph.adjacency_data(g) for g in crowds)


@gob.mapper(all_items=True)
def save_crowds(crowds):
    """
    save crowds to mongo
    """
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
    crowd_col = models.Crowd.database.Crowd
    crowd_col.ensure_index([('mloc','2d'),('zoom',1)],bits=20)
    crowd_col.ensure_index('zoom')


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
    """
    save users to mongo
    """
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
    """
    save tweets to mongo from streaming api files
    USAGE: zcat /spare/twitwatch/conv/*/*.gz | ./gb.py -s save_tweets
    """
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
        t.user_name = tweet['user']['name']
        t.user_loc = tweet['user'].get('location','')
        t.profile_image_url = tweet['user']['profile_image_url']
        t.ents = tweet['entities']
        t.save()
    # I don't entirely like ensuring the index here. It might be cleaner to make
    # it a separate command.
    tweet_col = models.Tweet.database.Tweet
    tweet_col.ensure_index('cid')


@gob.mapper(all_items=True)
def count_topics(crowds):
    """
    Analyze the topics a crowd is discussing.  (Currently only sorts between
    democratic and republican terms.)
    """
    #FIXME: don't hardcode political details here
    topics = dict(
        red = ('romney','mitt','gop'),
        blue = ('obama','barack','dnc','charlottein2012'),
    )
    for crowd_ in crowds:
        crowd_id = dict(crowd_['graph'])['id']
        crowd = models.Crowd.get_id(crowd_id)
        counts = collections.defaultdict(int)
        for tweet in models.Tweet.find({'cid':crowd_id}):
            for label,terms in topics.iteritems():
                text = tweet.text.lower()
                if any(term in text for term in terms):
                    counts[label]+=1
        crowd.topics = dict(counts)
        crowd.save()
