from itertools import chain
import networkx as nx
import datetime

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


