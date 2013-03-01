import time
import calendar
import collections

import numpy as np
from scipy.signal import convolve2d
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
    TILE_SIZE = .2
    def _msl_tile(deg):
        return int(round(deg/TILE_SIZE))

    frames = collections.defaultdict(lambda: np.zeros((int(360/TILE_SIZE),int(180/TILE_SIZE))))

    for stamp,(lng,lat) in tweet_locs:
        frame = stamp//60
        if frame<LANDING-180: # data for t-180
            continue
        if frame>=LANDING+180:
            break

        lng_tile = _msl_tile(lng) + int(180/TILE_SIZE)
        lat_tile = _msl_tile(lat) + int(90/TILE_SIZE)
        try:
            frames[frame][lng_tile,lat_tile]+=1
        except IndexError:
            pass

    for id,frame in frames.iteritems():
        filter = [[1,2,1],[2,4,2],[1,2,1]]
        frames[id] = convolve2d(frame,filter,mode="same")

    for id,frame in frames.iteritems():
        bef = frames.get(id-1)
        aft = frames.get(id+1)
        if bef is None or aft is None:
            continue
        scaled = np.log(bef+aft+2*frame+1)
        data = np.minimum(scaled, 5)*51
        buff = np.require(np.transpose(data),np.uint8,['C_CONTIGUOUS'])
        img = Image.frombuffer('L',(data.shape),buff)

        message = "%d minutes %s landing"%(
                abs(LANDING-id),
                "after" if id>LANDING else "before",
            )
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype('/Library/Fonts/Arial.ttf',36)
        draw.text((10, 10), message, font=font, fill=192)
        img.save("%d.png"%id)

