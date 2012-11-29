from base import gob


@gob.mapper(all_items=True)
def connected_ids(tweets):
    seen,ated = set(),set()
    for tweet in tweets:
        seen.add(tweet['user']['id'])
        for mention in tweet['entities']['user_mentions']:
            ated.add(mention['id'])
    return ated.union(seen)


@gob.mapper(slurp={'connected_ids':set})
def connected_users(tweets,connected_ids):
    seen = set()
    for tweet in tweets:
        if tweet['user']['id'] not in connected_ids:
            continue
        if tweet['user']['id'] in seen:
            continue
        yield tweet['user']

