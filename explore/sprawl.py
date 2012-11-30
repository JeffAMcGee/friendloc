#!/usr/bin/env python
import numpy
import random
import logging
import itertools
import collections

from base import gob
from base.models import Edges, User, Tweets
from base import twitter
from base.gisgraphy import GisgraphyResource
from collections import defaultdict
import base.utils as utils


NEBR_KEYS = ['rfriends','just_followers','just_friends','just_mentioned']


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


def _untangle_users_and_coords(users_and_coords):
    users = {}
    locs = defaultdict(list)
    for user_or_coord in users_and_coords:
        if isinstance(user_or_coord,dict):
            users[user_or_coord['id']] = user_or_coord
        else:
            uid,coord = user_or_coord
            locs[uid].append(coord)
    return users, locs


@gob.mapper(all_items=True)
def mloc_users(users_and_coords):
    users, locs = _untangle_users_and_coords(users_and_coords)
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


def _fetch_edges(twit,uid):
    edges = Edges.get_id(uid)
    if not edges:
        edges = twit.get_edges(uid)
        edges.save()
    return edges


def _fetch_tweets(twit,uid):
    tweets = Tweets.get_id(uid)
    if not tweets:
        tweets_ = twit.user_timeline(uid)
        tweets = Tweets(_id=uid,tweets=tweets_)
        tweets.save()
    return tweets


def _contact_sets(tweets, edges):
    ated = set(tweets.ats or [])
    frds = set(edges.friends)
    fols = set(edges.followers)
    return dict(
        rfriends = frds&fols,
        just_friends = frds-fols,
        just_followers = fols-frds,
        just_mentioned = ated-(frds|fols),
        )


def _pick_best_contacts(user, sets, limit=100):
    def digit_sum(uid):
        return sum(map(int,str(uid)))
    left = limit
    for key in NEBR_KEYS:
        if left>0:
            uids = sorted(sets[key],key=digit_sum,reverse=True)[:left]
        else:
            uids = []
        left -=len(uids)
        setattr(user,key,uids)


def _pick_random_contacts(user, sets, limit=100):

    #pick uids from sets
    for key,s in sets.iteritems():
        l = list(s)
        random.shuffle(l)
        setattr(user,key,l[:limit//4])


def _save_user_contacts(twit,user,contact_picker,limit):
    logging.info("visit %s - %d",user.screen_name,user._id)
    edges, tweets = None, None
    try:
        edges = _fetch_edges(twit,user._id)
        tweets = _fetch_tweets(twit,user._id)
        sets = _contact_sets(tweets,edges)
        contact_picker(user,sets,limit)
    except twitter.TwitterFailure as e:
        logging.warn("%d for %d",e.status_code,user._id)
        user.error_status = e.status_code
    user.save()
    return edges, tweets


def _my_contacts(user):
    return ((User.mod_id(c),c) for c in user.contacts)


@gob.mapper(all_items=True)
def find_contacts(user_ds):
    gis = GisgraphyResource()
    twit = twitter.TwitterResource()
    for user_d in itertools.islice(user_ds,2600):
        user = User.get_id(user_d['id'])
        if user:
            logging.warn("not revisiting %d",user._id)
        else:
            user = User(user_d)
            user.geonames_place = gis.twitter_loc(user.location)
            _save_user_contacts(twit, user, _pick_random_contacts, limit=100)
        for mod_nebr in _my_contacts(user):
            yield mod_nebr


@gob.mapper()
def find_leafs(uid):
    twit = twitter.TwitterResource()
    user = User.get_id(uid)
    _save_user_contacts(twit, user, _pick_random_contacts, limit=100)
    return _my_contacts(user)


@gob.mapper(all_items=True)
def mloc_uids(user_ds):
    retrieved = [u['id'] for u in itertools.islice(user_ds,2600)]
    users = User.find(User._id.is_in(retrieved))
    good_ = { u._id for u in users if any(getattr(u,k) for k in NEBR_KEYS)}
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


@gob.mapper(all_items=True,slurp={'mdists':next})
def lookup_contacts(contact_uids,mdists,env):
    twit = twitter.TwitterResource()
    gis = GisgraphyResource()
    gis.set_mdists(mdists)

    # FIXME: we need a better way to know which file we are on.
    # FIXME: use the new input_paths thing
    first, contact_uids = utils.peek(contact_uids)
    group = User.mod_id(first)
    logging.info('lookup old uids for %s',group)
    save_name = 'saved_users.%s'%group
    if env.name_exists(save_name):
        stored = set(env.load(save_name))
    else:
        stored = User.mod_id_set(int(group))
    logging.info('loaded mod_group %s of %d users',group,len(stored))
    missing = (id for id in contact_uids if id not in stored)

    chunks = utils.grouper(100, missing, dontfill=True)
    for chunk in chunks:
        users = twit.user_lookup(user_ids=list(chunk))
        for amigo in filter(None,users):
            assert User.mod_id(amigo._id)==group
            amigo.geonames_place = gis.twitter_loc(amigo.location)
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


@gob.mapper(all_items=True,slurp={'mdists':next})
def fix_mloc_mdists(mloc_uids,mdists):
    gis = GisgraphyResource()
    gis.set_mdists(mdists)
    # We didn't have mdists at the time the mloc users were saved. This
    # function could be avoided by running the mdist calculation before
    # running find_contacts.
    fixed = 0
    users = User.find(User._id.is_in(tuple(mloc_uids)))
    for user in users:
        user.geonames_place = gis.twitter_loc(user.location)
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


def _fetch_profiles(uids,twit,gis):
    users = list(User.find(User._id.is_in(uids)))
    existing_ids = {u._id for u in users}
    missing_ids = [uid for uid in uids if uid not in existing_ids]

    chunks = utils.grouper(100, missing_ids, dontfill=True)
    for chunk in chunks:
        found = twit.user_lookup(user_ids=list(chunk))
        for amigo in filter(None,found):
            amigo.geonames_place = gis.twitter_loc(amigo.location)
            amigo.merge()
            users.append(amigo)
    return users


def _calc_lorat(nebrs,twit,gis):
    leaf_ids = {uid
             for nebr in nebrs
             for uid in nebr.contacts[:10]}
    leafs_ = _fetch_profiles(list(leaf_ids),twit,gis)
    leafs = {leaf._id:leaf for leaf in leafs_}

    for nebr in nebrs:
        # Does this break if the contact does not exist?
        nebr_loc = nebr.geonames_place.to_d()
        dists = []
        for leaf_id in nebr.contacts[:10]:
            leaf = leafs.get(leaf_id)
            if leaf and leaf.has_place():
                dist = utils.coord_in_miles(nebr_loc,leaf.geonames_place.to_d())
                dists.append(dist)
        if dists:
            lorat = sum(1.0 for d in dists if d<25)/len(dists)
        else:
            lorat = float('nan')
        nebr.local_ratio = lorat


CrawlResults = collections.namedtuple("CrawlResults",['nebrs','ats','ated'])


def crawl_single(user, twit, gis):
    """
    save a single user, contacts, and leafs to the database

    crawl a user object who has not been visited before
    twit is a TwitterResource
    gis is a GisgraphyResource with mdists
    """

    edges,tweets=_save_user_contacts(twit, user, _pick_best_contacts, limit=100)
    contact_ids = user.contacts
    profiles = {p._id:p for p in _fetch_profiles(contact_ids,twit,gis)}

    def has_place(uid):
        return uid in profiles and profiles[uid].has_place()
    user.neighbors = filter(has_place, contact_ids)[:25]
    nebrs = [profiles[nid] for nid in user.neighbors]

    ated = set()
    for nebr in nebrs:
        ne,nt = _save_user_contacts(twit, nebr, _pick_best_contacts, limit=100)
        if nt and nt.ats and user._id in nt.ats:
            ated.add(nebr._id)

    need_lorat = [nebr for nebr in nebrs if nebr.local_ratio is None]
    _calc_lorat(need_lorat,twit,gis)
    for nebr in need_lorat:
        nebr.merge()
    user.save()

    return CrawlResults(nebrs,tweets.ats if tweets else [],ated)

