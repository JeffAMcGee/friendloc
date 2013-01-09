from itertools import chain
import networkx as nx

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
def connected_ats(tweets,connected_ids):
    # pick out the time of mentions between connected users
    for tweet in tweets:
        uid = tweet['user']['id']
        if uid not in connected_ids:
            continue
        for mention in tweet['entities']['user_mentions']:
            if mention['id'] in connected_ids:
                yield uid,mention['id'],tweet['created_at']

