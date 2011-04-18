#!/usr/bin/env python
import sys
import numpy
import random
import itertools
import logging

import maroon
from restkit.errors import Unauthorized, ResourceNotFound

from base.models import Edges, User, Tweets
from base.twitter import TwitterResource
from base.gisgraphy import GisgraphyResource
from base.splitproc import SplitProcess
import base.utils as utils
from settings import settings


class GeoLookup(SplitProcess):
    def __init__(self,path,db_name,**kwargs):
        SplitProcess.__init__(self, **kwargs)
        self.path = path
        self.db_name = db_name

    def produce(self):
        users = {}
        for i,t in enumerate(utils.read_json(self.path)):
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
            if numpy.median(dists)>50: continue
            del user['locs']
            user['mloc'] = median
            yield user

    def map(self,users):
        self.twitter = TwitterResource()
        self.gis = GisgraphyResource()
        maroon.Model.database = maroon.MongoDB(
                host=settings.db_host,
                name=self.db_name)

        for user_d in users:
            self.twitter.sleep_if_needed()
            user = User.get_id(user_d['id'])
            if user:
                user.update(user_d)
            else:
                user = User(user_d)
            logging.info("visit %s - %d",user.screen_name,user._id)
            try:
                self.save_neighbors(user)
                user.merge()
            except ResourceNotFound:
                logging.warn("ResourceNotFound for %d",user._id)
            except Unauthorized:
                logging.warn("Unauthorized for %d",user._id)
            yield None

    def save_neighbors(self,user):
        self.save_user_data(user._id)
        edges = Edges.get_id(user._id)
        tweets = Tweets.get_id(user._id,fields=['ats'])
        ated = set(tweets.ats)
        frds = set(edges.friends)
        fols = set(edges.followers)
        sets = dict(
            rfriends = frds&fols,
            just_friends = frds-fols,
            just_followers = fols-frds,
            just_mentioned = ated-(frds|fols),
            )
        #pick 100 uids from sets
        for k,s in sets.iteritems():
            l = list(s)
            random.shuffle(l)
            sets[k] = l
        uids = itertools.islice(
            itertools.ifilter(lambda x: x is not None,
                itertools.chain.from_iterable(
                    itertools.izip_longest(
                        *sets.values()))),
            0, 100)
        users = self.twitter.user_lookup(user_ids=list(uids))

        amigos = []
        for amigo in users:
            if not amigo or amigo.protected: continue
            place = self.gis.twitter_loc(amigo.location)
            #ignore states and regions with more than 5 million people
            if (    not place
                    or place.feature_code=="ADM1"
                    or place.population>5000000):
                continue
            amigo.geonames_place = place
            amigos.append(amigo)
        for amigo in amigos:
            amigo.merge()
        amigo_ids = set(a._id for a in amigos)

        save_limit = 12
        for k,s in sets.iteritems():
            group = [aid for aid in s if aid in amigo_ids]
            if group:
                setattr(user,k,group)
                if self.save_user_data(group[0]):
                    save_limit-=1
        for k in ['rfriends','just_followers','just_friends','just_mentioned']:
            group = getattr(user,k) or []
            for uid in group[1:]:
                if self.save_user_data(uid):
                    save_limit-=1
                if save_limit==0:
                    return

    def save_user_data(self,uid):
        try:
            if not Edges.in_db(uid):
                edges = self.twitter.get_edges(uid)
                edges.save()

            if not Tweets.in_db(uid):
                tweets = self.twitter.user_timeline(uid)
                Tweets(_id=uid,tweets=tweets).save()
            return True
        except Unauthorized:
            logging.warn("Unauthorized for %d",uid)
            return False


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv)>1 else None
    proc = GeoLookup(path, "geo", log_level=logging.INFO, slaves=12)
    #proc.run_single()
    proc.run()
