#!/usr/bin/env python
import sys
sys.path.append("..")

import random
import itertools
import logging

from base.models import Edges, User, Tweets
from base.twitter import TwitterResource
from base.gisgraphy import GisgraphyResource
import base.utils as utils
from settings import settings
from base.splitproc import SplitProcess


class UsersToLookupFinder(SplitProcess):
    def __init__(self,db_name,**kwargs):
        SplitProcess.__init__(self, **kwargs)
        self.db_name = db_name
        self.chunk_lookup = _ChunckLookup(**kwargs)
        self.chunk_lookup.db_name = db_name

    def produce(self):
        return User.find(User.median_loc.exists(), fields=[])

    def startup(self):
        self.gis = GisgraphyResource()
        utils.use_mongo(self.db_name)
 
    def map(self,users):
        for user in users:
            logging.info("investigating %d",user._id)
            edges = Edges.get_id(user._id)
            tweets = Tweets.get_id(user._id,fields=['ats'])
            uids = set(itertools.chain(
                edges.friends,
                edges.followers,
                tweets.mentioned))
            if len(uids)>2000:
                chosen = random.sample(uids,2000)
                user = User.get_id(uid)
                user.many_edges = True
                user.save()
                edges.lookups = chosen
                edges.save()
                yield chosen
            else:
                yield uids

    def _filter_old_uids(self, uid_sets):
        logging.info("began to read old uids")
        done = set(u._id for u in User.find(fields=[]))
        logging.info("read old uids")
        for s in uid_sets:
            new = s-done
            done.update(new)
            logging.info("queueing %d of %d users",len(new),len(s))
            yield new

    def consume(self, uid_sets):
        self.chunk_lookup.groups = utils.grouper(100,
            itertools.chain.from_iterable(
                self._filter_old_uids(uid_sets)
            ))
        self.chunk_lookup.run()


class _ChunckLookup(SplitProcess):
    def startup(self):
        self.twitter = TwitterResource()
        self.gis = GisgraphyResource()
        utils.use_mongo(self.db_name)

    def produce(self):
        return self.groups
#SLEEP IF NEEDED
    def map(self, chunks):
        for chunk in chunks:
            users = filter(None,twitter.user_lookup(user_ids=list(uids)))
            saved = 0
            for amigo in filter(None,users):
                if not amigo or amigo.protected: continue
                place = self.gis.twitter_loc(amigo.location)
                if not place:
                    continue
                amigo.geonames_place = place
                amigo.merge()
                saved +=1
            logging.info("saved %d of %d",saved,len(users))
            yield None


if __name__ == '__main__':
    proc = UsersToLookupFinder("usgeo",
            label='geolookup',
            log_level=logging.INFO,
            slaves=8,
            debug=True)
    settings.pdb()
    proc.run()
