#!/usr/bin/env python
import sys
import numpy
import random
import itertools
import logging

import maroon
from restkit.errors import Unauthorized, ResourceNotFound

from localcrawl.models import Edges, User, Tweet
from localcrawl.twitter import TwitterResource
from localcrawl.gisgraphy import GisgraphyResource

from splitproc import SplitProcess
import utils


class GeoLookup(SplitProcess):
    def __init__(self,path,db_name,**kwargs):
        SplitProcess.__init__(self, **kwargs)
        self.path = path
        self.db_name = db_name

    def produce(self):
        users = {}
        db = maroon.MongoDB(name=self.db_name)
        old_users = db.User.find({'mloc':{'$exists':1}}, {'_id':1})
        old_uids = set(user['_id'] for user in old_users)
        logging.info("old_uids %d",len(old_uids))
        for i,t in enumerate(utils.read_json(self.path)):
            if i%10000 ==0:
                logging.info("read %d tweets"%i)
            uid = t['user']['id']
            if 'coordinates' not in t or uid in old_uids: continue
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
        maroon.Model.database = maroon.TeeDB(
            'logs/geocrawl_%d.json'%self.slave_id,
            maroon.MongoDB(name=self.db_name)
            )

        for user_d in users:
            self.twitter.sleep_if_needed()
            user = User(from_dict=user_d)
            try:
                self.save_neighbors(user)
                user.attempt_save()
            except ResourceNotFound:
                logging.info("ResourceNotFound for %d",user._id)
            except Unauthorized:
                logging.info("Unauthorized for %d",user._id)
            yield None

    def save_neighbors(self,user):
        edges, tweets = self.save_user_data(user._id)
        frds = set(edges.friends)
        fols = set(edges.followers)
        sets = dict(
            rfriends = frds&fols,
            just_friends = frds-fols,
            just_followers = fols-frds,
            )
        for k,s in sets.iteritems():
            if len(s)>33:
                sets[k] = set(random.sample(s,33))
        uids = list(itertools.chain(*sets.values()))
        users = self.twitter.user_lookup(user_ids=uids)
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
        User.database.bulk_save_models(amigos)

        #pick a [friend,rfriend,follower] and store their info
        for k,s in sets.iteritems():

            saved = s.intersection(a._id for a in amigos)
            if saved:
                group = list(saved)
                random.shuffle(group)
                setattr(user,k,group)
                self.save_user_data(group[0])

    def save_user_data(self,uid):
        edges = self.twitter.get_edges(uid)
        edges.attempt_save()

        tweets = self.twitter.user_timeline(uid)
        Tweet.database.bulk_save_models(tweets)
        return edges, tweets


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv)>1 else None
    proc = GeoLookup(path, "geo", log_level=logging.INFO)
    proc.run()
