import time
import calendar
import collections

import numpy as np
from scipy import sparse
import ImageFont, ImageDraw, Image

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


@gob.mapper(all_items=True)
def msl_pngs(tweet_locs):

    #landed  10:32 pdt -> 5:32 am utc
    LANDING = 22403852
    WIDTH = 640
    HEIGHT = 360
    TILE_WIDTH = 360.0/WIDTH
    TILE_HEIGHT = 180.0/HEIGHT
    SHAPE = (WIDTH,HEIGHT)

    def _msl_tile(loc):
        return (
            int(round((loc[0]+180)/TILE_WIDTH)),
            int(round((loc[1]+90)/TILE_HEIGHT)),
        )

    tweets = collections.defaultdict(list)
    for stamp,loc in tweet_locs:
        minute = stamp//60
        if minute<LANDING-180: continue
        if minute>=LANDING+180: break
        tweets[minute].append(loc)

    count = sum(len(locs) for locs in tweets.itervalues())
    print "loaded %d tweets"% count
    frames = dict()

    for minute,locs in tweets.iteritems():
        tiles = [_msl_tile(loc) for loc in locs]
        counts = collections.Counter(tiles)
        density = np.sqrt([counts[tile] for tile in tiles])
        rand_angles = np.pi*2*np.random.random(len(locs))
        rand_dists = np.random.random(len(locs))*density/5
        lngs,lats = zip(*locs)
        new_lngs = lngs + rand_dists*np.sin(rand_angles)*TILE_WIDTH
        new_lats = lats + rand_dists*np.cos(rand_angles)*TILE_HEIGHT

        frame = sparse.lil_matrix(SHAPE,dtype=np.int16)
        for i in xrange(len(locs)):
            new_tile = _msl_tile((new_lngs[i],new_lats[i]))
            frame[new_tile]+=1
        frames[minute] = frame
    print "made frames"

    first_frame = min(frames.iterkeys())+1

    for id,frame in frames.iteritems():
        bef = frames.get(id-1)
        aft = frames.get(id+1)
        if bef is None or aft is None:
            continue
        scaled = bef+aft+2*frame
        dense = np.array(scaled.todense())
        clipped = np.minimum(dense,7)
        nonzero = np.minimum(dense,1)
        data = clipped*32+nonzero*31
        buff = np.require(np.transpose(data),np.uint8,['C_CONTIGUOUS'])
        img = Image.frombuffer('L',(data.shape),buff)

        message = "%d minutes %s landing"%(
                abs(LANDING-id),
                "after" if id>LANDING else "before",
            )
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype('/Library/Fonts/Arial.ttf',18)
        draw.text((10, 10), message, font=font, fill=192)
        img.save("msl%03d.png"%(id-first_frame))
