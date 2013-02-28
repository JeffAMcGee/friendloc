import time
import calendar

from friendloc.base import gob, models

TWITTER_TIME_FORMAT="%a %b %d %H:%M:%S +0000 %Y"

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


@gob.mapper(all_items=True)
def msl_id_locs(msl_locs):
    """create mapping from user id to location"""
    for user in msl_locs:
        if user.get('ploc'):
            yield user['id'],user['ploc']


@gob.mapper(all_items=True,slurp={'msl_id_locs':dict})
def msl_tweet_locs(tweets, msl_id_locs):
    """create mapping from user id to location"""
    for tweet in tweets:
        uid = tweet['user']['id']
        if uid not in msl_id_locs:
            continue
        timetuple = time.strptime(tweet['created_at'],TWITTER_TIME_FORMAT)
        timestamp = calendar.timegm(timetuple)
        yield timestamp, msl_id_locs[uid]

