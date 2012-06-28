#!/usr/bin/env python
import numpy
import random
import logging
import itertools
from itertools import chain

from restkit import errors

from base import gob
from base.models import Edges, User, Tweets
from base import twitter
from base.gisgraphy import GisgraphyResource
from collections import defaultdict
import base.utils as utils


NEBR_KEYS = ['rfriends','just_followers','just_friends','just_mentioned']


class Sprawler(object):
    def __init__(self,env):
        self.twitter = twitter.TwitterResource()
        self.gis = GisgraphyResource()
        try:
            mdists = next(env.load('mdists'))
        except (IOError,KeyError):
            pass
        else:
            self.gis.set_mdists(mdists)


@gob.mapper(all_items=True)
def parse_geotweets(tweets):
    # USAGE:
    # gunzip -c ~/may/*/*.gz | ./gb.py -s parse_geotweets
    users = set()
    for i,t in enumerate(tweets):
        if i%10000 ==0:
            logging.info("read %d tweets"%i)
        if 'id' not in t: continue # this is not a tweet
        uid = t['user']['id']
        if not t.get('coordinates'): continue
        if uid not in users:
            yield User.mod_id(uid),t['user']
            users.add(uid)
        yield User.mod_id(uid),(uid,t['coordinates']['coordinates'])
    logging.info("sending up to %d users"%len(users))


@gob.mapper(all_items=True)
def mloc_uids(user_ds):
    retrieved = [u['id'] for u in itertools.islice(user_ds,2600)]
    # this query is probably painful.
    users = User.find(User._id.is_in(retrieved) and User.neighbors.exists())
    good_ = { u._id for u in users }
    good = [uid for uid in retrieved if uid in good_]
    # throw away accounts that didn't work to get down to the 2500 good users
    return good[:2500]


@gob.mapper(all_items=True)
def mloc_users(users_and_coords):
    users = {}
    locs = defaultdict(list)
    for user_or_coord in users_and_coords:
        if isinstance(user_or_coord,dict):
            users[user_or_coord['id']] = user_or_coord
        else:
            uid,coord = user_or_coord
            locs[uid].append(coord)
    selected = []
    for uid,user in users.iteritems():
        spots = locs[uid]
        if len(spots)<=2: continue
        if user['followers_count']==0 and user['friends_count']==0: continue
        median = utils.median_2d(spots)
        dists = [utils.coord_in_miles(median,spot) for spot in spots]
        if numpy.median(dists)>50:
            continue #user moves too much
        user['mloc'] = median
        selected.append(user)
    random.shuffle(selected)
    return selected


class EdgeFinder(Sprawler):
    def _pick_contacts(self, user, limit):
        edges = Edges.get_id(user._id)
        if not edges:
            edges = self.twitter.get_edges(user._id)
            edges.save()

        tweets = Tweets.get_id(user._id)
        if not tweets:
            tweets_ = self.twitter.user_timeline(user._id)
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
            setattr(user,key,l[:limit])


    def _save_user_contacts(self,user,limit):
        logging.info("visit %s - %d",user.screen_name,user._id)
        try:
            self._pick_contacts(user,limit)
        except errors.ResourceError as e:
            logging.warn("%d for %d",e.status_int,user._id)
            user.error_status = e.status_int
        user.save()

    def _my_nebrs(self,user):
        nebrs = chain.from_iterable(
                    getattr(user,key) or ()
                    for key in NEBR_KEYS
                    )
        return ((User.mod_id(nebr),nebr) for nebr in nebrs)

    @gob.mapper(all_items=True)
    def find_contacts(self,user_ds):
        for user_d in itertools.islice(user_ds,2600):
            user = User.get_id(user_d['id'])
            if user:
                logging.warn("not revisiting %d",user._id)
            else:
                user = User(user_d)
                user.geonames_place = self.gis.twitter_loc(user.location)
                self._save_user_contacts(user,limit=25)
            for mod_nebr in self._my_nebrs(user):
                yield mod_nebr

    @gob.mapper()
    def find_leafs(self,uid):
        user = User.get_id(uid)
        self._save_user_contacts(user,limit=25)
        return self._my_nebrs(user)


@gob.mapper(all_items=True)
def contact_split(groups):
    visited = set(u._id for u in User.find(fields=[]))
    for group,ids in groups:
        for id in ids:
            if id not in visited:
                yield group,id


class ContactLookup(Sprawler):
    @gob.mapper(all_items=True)
    def lookup_contacts(self,contact_uids):
        assert self.gis._mdist
        chunks = utils.grouper(100, contact_uids)
        for chunk in chunks:
            users = self.twitter.user_lookup(user_ids=list(chunk))
            for amigo in filter(None,users):
                amigo.geonames_place = self.gis.twitter_loc(amigo.location)
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
