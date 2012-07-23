#!/usr/bin/env python
import numpy
import random
import logging
import itertools

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
        self.env = env
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
        except twitter.TwitterFailure as e:
            logging.warn("%d for %d",e.status_code,user._id)
            user.error_status = e.status_code
        user.save()

    def _my_contacts(self,user):
        return ((User.mod_id(c),c) for c in user.contacts)

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
            for mod_nebr in self._my_contacts(user):
                yield mod_nebr

    @gob.mapper()
    def find_leafs(self,uid):
        user = User.get_id(uid)
        self._save_user_contacts(user,limit=25)
        return self._my_contacts(user)


@gob.mapper(all_items=True)
def mloc_uids(user_ds):
    retrieved = [u['id'] for u in itertools.islice(user_ds,2600)]
    users = User.find(User._id.is_in(retrieved))
    good_ = set( u._id for u in users if any(getattr(u,k) for k in NEBR_KEYS))
    good = [uid for uid in retrieved if uid in good_]
    logging.info("found %d of %d",len(good),len(retrieved))
    # throw away accounts that didn't work to get down to the 2500 good users
    return good[:2500]


@gob.mapper(all_items=True)
def trash_extra_mloc(mloc_uids):
    "remove the mloc_users that mloc_uids skipped over"
    # This scares me a bit, but it's too late to go back and fix find_contacts.
    # I really wish I had limited find_contacts to stop after 2500 good users.
    db = User.database
    mloc_uids = set(mloc_uids)
    group_ = set(uid%100 for uid in mloc_uids)
    assert len(group_)==1
    group = next(iter(group_))
    stored = User.mod_id_set(group)
    trash = list(stored - mloc_uids)
    logging.info("trashing %d users",len(trash))
    logging.debug("full list: %r",trash)
    db.Edges.remove({'_id':{'$in':trash}})
    db.Tweets.remove({'_id':{'$in':trash}})
    db.User.remove({'_id':{'$in':trash}})


@gob.mapper(all_items=True)
def contact_split(groups):
    # FIXME this is now identical to nebr_split
    for group,ids in groups:
        for id in ids:
            yield group,id


@gob.mapper()
def saved_users():
    users = User.database.User.find({},fields=[],timeout=False)
    return ((User.mod_id(u['_id']),u['_id']) for u in users)


class ContactLookup(Sprawler):
    @gob.mapper(all_items=True)
    def lookup_contacts(self,contact_uids):
        assert self.gis._mdist

        # FIXME: we need a better way to know which file we are on.
        first, contact_uids = utils.peek(contact_uids)
        group = User.mod_id(first)
        logging.info('lookup old uids for %s',group)
        save_name = 'saved_users.%s'%group
        if self.env.name_exists(save_name):
            stored = set(self.env.load(save_name))
        else:
            stored = User.mod_id_set(int(group))
        logging.info('loaded mod_group %s of %d users',group,len(stored))
        missing = (id for id in contact_uids if id not in stored)

        chunks = utils.grouper(100, missing, dontfill=True)
        for chunk in chunks:
            users = self.twitter.user_lookup(user_ids=list(chunk))
            for amigo in filter(None,users):
                assert User.mod_id(amigo._id)==group
                amigo.geonames_place = self.gis.twitter_loc(amigo.location)
                amigo.merge()
            yield len(users)


def _pick_neighbors(user):
    nebrs = {}
    for key in NEBR_KEYS:
        cids = getattr(user,key)
        if not cids:
            continue

        # this is slowish
        contacts = User.find(User._id.is_in(cids), fields=['gnp'])
        nebrs[key] = set(u._id for u in contacts if u.has_place())

    picked_ = filter(None,
                itertools.chain.from_iterable(
                    itertools.izip_longest(*nebrs.values())))
    picked = picked_[:25]
    logging.info('picked %d of %d contacts',len(picked),len(user.contacts))
    return picked


@gob.mapper()
def pick_nebrs(mloc_uid):
    # reads predict.prep.mloc_uids, requires lookup_contacts, but don't read it.
    user = User.get_id(mloc_uid)
    user.neighbors = _pick_neighbors(user)
    user.save()
    return ((User.mod_id(n),n) for n in user.neighbors)


class MDistFixer(Sprawler):
    @gob.mapper(all_items=True)
    def fix_mloc_mdists(self,mloc_uids):
        # We didn't have mdists at the time the mloc users were saved. This
        # function could be avoided by running the mdist calculation before
        # running find_contacts.
        assert self.gis._mdist
        fixed = 0
        users = User.find(User._id.is_in(tuple(mloc_uids)))
        for user in users:
            user.geonames_place = self.gis.twitter_loc(user.location)
            user.save()
            if user.geonames_place:
                fixed+=1
        logging.info("fixed %d mdists",fixed)
        return [fixed]


@gob.mapper(all_items=True)
def nebr_split(groups):
    # This method should really be built into gob somehow.
    return (
        (group, id)
        for group,ids in groups
        for id in ids
        )
