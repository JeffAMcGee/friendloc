
from friendloc.base import gob, models


@gob.mapper(all_items=True)
def msl_users(tweets):
    """
    read in tweets and return user dicts for user ids in connected_ids
    USAGE: zcat data/mars_tweets.json.gz | ./gb.py -s msl_users
    """
    seen = set()
    for tweet in tweets:
        uid = tweet['user']['id']
        if uid in seen:
            continue
        seen.add(uid)
        yield models.User.mod_id(uid),tweet['user']
