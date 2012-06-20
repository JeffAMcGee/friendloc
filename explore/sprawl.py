#!/usr/bin/env python
import numpy
import random
import logging
from itertools import chain

from restkit.errors import Unauthorized, ResourceNotFound

from base import gob
from base.models import Edges, User, Tweets
from base import twitter
from base.gisgraphy import GisgraphyResource
import base.utils as utils


NEBR_KEYS = ['rfriends','just_followers','just_friends','just_mentioned']


@gob.mapper(all_items=True)
def mloc_users(tweets):
    # FIXME: how to read gzipped json?
    users = {}
    for i,t in enumerate(tweets):
        if i%10000 ==0:
            logging.info("read %d tweets"%i)
        if 'id' not in t: continue # this is not a tweet
        uid = t['user']['id']
        if not t.get('coordinates'): continue
        if uid not in users:
            users[uid] = t['user']
            users[uid]['locs'] = []
        users[uid]['locs'].append(t['coordinates']['coordinates'])
    logging.info("sending up to %d users"%len(users))
    for uid,user in users.iteritems():
        logging.debug("considering uid %d"%uid)
        spots = user['locs']
        if len(spots)<2: continue
        if user['followers_count']==0 and user['friends_count']==0: continue
        median = utils.median_2d(spots)
        dists = [utils.coord_in_miles(median,spot) for spot in spots]
        #if not (-126<median[0]<-66 and 24<median[1]<50):
        #    continue #not in us
        if numpy.median(dists)>50:
            continue #user moves too much
        del user['locs']
        user['mloc'] = median
        yield User.mod_id(user),user


def _save_user_contacts(twitter,user):
    edges = Edges.get_id(user._id)
    if not edges:
        edges = twitter.get_edges(user._id)
        edges.save()

    tweets = Tweets.get_id(user._id)
    if not tweets:
        tweets_ = twitter.user_timeline(user._id)
        tweets = Tweets(_id=user._id,tweets=tweets_)
        tweets.save()

    ated = set(tweets.ats or [])
    frds = set(edges.friends)
    fols = set(edges.followers)
    sets = dict(
        rfriends = frds&fols,
        just_friends = frds-fols,
        just_followers = fols-frds,
        just_mentioned = ated-(frds|fols),
        )

    #pick uids from sets
    for key,s in sets.iteritems():
        l = list(s)
        random.shuffle(l)
        setattr(user,key,l[:50])


class EdgeFinder():
    def __init__(self,env):
        self.twitter = twitter.TwitterResource()
        self.gis = GisgraphyResource()

    @gob.mapper()
    def find_edges(self,user_d):
        self.twitter.sleep_if_needed()
        user = User.get_id(user_d['id'])
        if user:
            # FIXME: need better anti-revisit code
            if user.neighbors:
                logging.warn("not revisiting %d",user._id)
                return ()
            user.update(user_d)
        else:
            user = User(user_d)
            user.geonames_place = self.gis.twitter_loc(user.location)
        logging.info("visit %s - %d",user.screen_name,user._id)
        try:
            _save_user_contacts(self.twitter,user)
            user.merge()
        except ResourceNotFound:
            logging.warn("ResourceNotFound for %d",user._id)
            return ()
        except Unauthorized:
            logging.warn("Unauthorized for %d",user._id)
            return ()

        nebrs = chain.from_iterable(getattr(user,key) for key in NEBR_KEYS)
        return ((User.mod_id(nebr),nebr) for nebr in nebrs)


@gob.mapper(all_items=True)
def contact_split(groups):
    visited = set(u._id for u in User.find(fields=[]))
    for group,ids in groups:
        for id in ids:
            if id not in visited:
                yield group,id


@gob.mapper(all_items=True)
def lookup_contacts(contact_uids):
    twtr = twitter.TwitterResource()
    gis = GisgraphyResource()
    chunks = utils.grouper(100, contact_uids)

    for chunk in chunks:
        twtr.sleep_if_needed()
        users = filter(None,twtr.user_lookup(user_ids=list(chunk)))
        for amigo in users:
            amigo.geonames_place = gis.twitter_loc(amigo.location)
            amigo.merge()
        yield len(users)


def _has_place(user):
    gnp = user.geonames_place
    return gnp and gnp.mdist<1000 and gnp.population<10**7


def _pick_neighbors(user):
    nebrs = {}
    for key in NEBR_KEYS:
        cids = getattr(user,key)
        # this is slowish
        contacts = User.find(User._id.is_in(cids), fields=['gnp'])
        nebrs[key] = set(u['_id'] for u in contacts if _has_place(u))
    return nebrs


# read predict.prep.mloc_uids, require lookup_contacts, but don't read it.
def pick_nebrs(mloc_uid):
    user = User.get_id(mloc_uid)
    if not user.neighbors:
        user.neigbors = _pick_neighbors(user)
        user.save()
    return user.neighbors

# set_reduce nebrs and split by nebr_uid

def find_leaves(nebr):
    Edges.fetch_and_save()
    Tweets.fetch_and_save()
    nebr.save()
    for leaf_uid in (Edges and Tweets)[:400]:
        yield leaf_uid

# set_reduce leafs and split by nebr_uid
def lookup_leaves(leaf_uid):
    # do this in groups of 100
    user = User.lookup(leaf_uid)
    user.gisgraphy()
    user.save()
