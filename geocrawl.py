#!/usr/bin/env python
import sys
import numpy
import random
import itertools

import maroon
from restkit.errors import Unauthorized

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
        for t in utils.read_json(self.path):
            uid = t['user']['id']
            if 'coordinates' not in t: continue
            if uid not in users:
                users[uid] = t['user']
                users[uid]['locs'] = []
            users[uid]['locs'].append(t['coordinates']['coordinates'])
        for uid,user in users.iteritems():
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
        maroon.Model.database = maroon.MongoDB(name=self.db_name)

        for user_d in users:
            self.twitter.sleep_if_needed()
            user = User(from_dict=user_d)
            try:
                self.save_neighbors(user)
                user.attempt_save()
            except Unauthorized:
                pass

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
            place = self.gis.twitter_loc(amigo.location)
            #ignore states and regions with more than 5 million people
            if( amigo.protected
                    or not place
                    or place.feature_code=="ADM1"
                    or place.population>5000000):
                for k,s in sets.iteritems():
                    s.discard(amigo._id)
                continue
            amigo.geonames_place = place
            amigos.append(amigo)
        User.database.bulk_save_models(amigos)

        #pick a [friend,rfriend,follower] and store their info
        for k,s in sets.iteritems():
            if s:
                group = list(s)
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
    proc = GeoLookup(sys.argv[0])
    proc.run_single()
