#!/usr/bin/env python
from collections import defaultdict
try:
    import beanstalkc
except:
    pass
import json
import pdb
import signal
from datetime import datetime
import time
from itertools import groupby
import logging
import sys

import maroon
from maroon import *
from restkit import ResourceNotFound

from settings import settings
from base.models import Edges, User, Tweet, LookupJobBody
from base.twitter import TwitterResource
from base.gisgraphy import GisgraphyResource, in_local_box
from procs import LocalProc, create_slaves
from scoredict import Scores, log_score, BUCKETS
import scoredict


RFRIEND_POINTS = 1000
MENTION_POINTS = 10


class LookupMaster(LocalProc):
    def __init__(self):
        LocalProc.__init__(self,"lookup")
        self.scores = Scores()
        if settings.lookup_in:
            self.scores.read(settings.lookup_in)
        self.lookups = self.scores.count_lookups()
        self.halt = False

    def run(self):
        print "starting lookup"
        logging.info("started lookup")
        try:
            while not self.halt:
                tube = self.stalk.using()
                ready = self.stalk.stats_tube(tube)['current-jobs-ready']
                logging.info("ready is %d",ready)
                if ready<1000:
                    cutoff = self.calc_cutoff()
                    old_lookups = self.lookups

                    if cutoff<settings.min_cutoff and self.lookups>100:
                        self.pick_users(settings.min_cutoff)
                    else:
                        self.pick_users(cutoff)
                    print "scores:%d lookups:%d"%(len(self.scores),self.lookups)
                    if old_lookups == self.lookups:
                        print "halt because no new lookups"
                        self.halt=True
                        self.read_scores()
                        self.force_lookup()

                logging.info("read_scores")
                self.read_scores()
        except:
            logging.exception("exception caused HALT")
        self.read_scores()
        self.scores.dump(settings.lookup_out)
        print "Lookup is done!"

    def read_scores(self):
        job = None
        stop = 10000000 if self.halt else 100000
        for x in xrange(stop):
            try:
                job = self.stalk.reserve(35)
                if job is None:
                    logging.info("loaded %d scores",x)
                    return
                if job.body=="halt":
                    self.halt=True
                    print "starting to halt..."
                    logging.info("starting to halt...")
                    job.delete()
                    return
                body = LookupJobBody.from_job(job)
                if body.done:
                    self.scores.set_state(body._id, scoredict.DONE)
                else:
                    self.scores.increment(
                        body._id,
                        body.rfriends_score,
                        body.mention_score
                    )
                job.delete()
            except:
                logging.exception("exception in read_scores caused HALT")
                self.halt = True
                if job:
                    job.bury()
                return

    def calc_cutoff(self):
        self.stats = [0 for x in xrange(BUCKETS)]
        for u in self.scores:
            state, rfs, ats = self.scores.split(u)
            if state==scoredict.NEW:
                self.stats[log_score(rfs,ats)]+=1
        for count,score in zip(self.stats,xrange(BUCKETS)):
            logging.info("%d %d",score,count)
        total = 0
        for i in xrange(BUCKETS-1,-1,-1):
            total+=self.stats[i]
            if total > settings.crawl_ratio*(len(self.scores)-self.lookups):
                return i
        return 0

    def pick_users(self, cutoff):
        logging.info("pick_users with score %d", cutoff)
        for uid in self.scores:
            state, rfs, ats = self.scores.split(uid)
            if state==scoredict.NEW and log_score(rfs,ats) >= cutoff:
                self._send_job(uid,rfs,ats)
                self.lookups+=1

    def _send_job(self, uid, rfs, ats, force=None):
        job = LookupJobBody(
            _id=uid,
            rfriends_score=rfs,
            mention_score=ats,
            force=force
        )
        job.put(self.stalk)
        self.scores.set_state(uid, scoredict.LOOKUP)

    def force_lookup(self):
        "Lookup users who were not included in the original crawl."
        for user in User.get_all():
            if (    user.lookup_done or
                    user.protected or
                    user._id not in self.scores or
                    user.local_prob==1
               ):
                continue

            state, rfs, ats = self.scores.split(user._id)
            reasons = [
                user.utc_offset == settings.utc_offset,
                log_score(rfs,ats) >= settings.non_local_cutoff,
                user.local_prob == .5,
            ]
            if sum(reasons)>=2:
                logging.info("force %s - %d for %r", user.screen_name, user._id, reasons)
                self._send_job(user._id,rfs,ats,True)


def guess_location(user, gisgraphy):
    if not user.location:
        return .5
    place = gisgraphy.twitter_loc(user.location)
    if not place:
        return .5
    user.geonames_place = place
    return 1 if in_local_box(place.to_d()) else 0


class LookupSlave(LocalProc):
    def __init__(self,slave_id):
        LocalProc.__init__(self,'lookup',slave_id)
        self.twitter = TwitterResource()
        self.gisgraphy = GisgraphyResource()

    def run(self):
        while True:
            jobs = []
            for x in xrange(100):
                try:
                    # reserve blocks to wait when x is 0, but returns None for 1-99
                    j = self.stalk.reserve(0 if x else None)
                except beanstalkc.DeadlineSoon:
                    break
                if j is None:
                    break
                jobs.append(j)

            bodies = [LookupJobBody.from_job(j) for j in jobs]
            try:
                users =self.twitter.user_lookup([b._id for b in bodies])
            except ResourceNotFound:
                logging.info("no profile for %r",[b._id for b in bodies])
                continue

            logging.info("looking at %r"%[getattr(u,'screen_name','') for u in users])
            for job,body,user in zip(jobs,bodies,users):
                if user is None:
                    logging.info("no profile for %d",body._id)
                    job.delete()
                    continue
                try:
                    self.twitter.sleep_if_needed()
                    logging.info("look at %s",user.screen_name)
                    if (not body.force) and User.in_db(user._id):
                        job.delete()
                        continue
                    self.crawl_user(user,body.force)
                    user.save()
                    job.delete()
                except:
                    logging.exception("exception for job %s"%job.body)
                    job.bury()
            logging.info("api calls remaining: %d",self.twitter.remaining)

    def crawl_user(self,user,force):
        user.local_prob = guess_location(user,self.gisgraphy)
        if (user.local_prob != 1.0 and not force) or user.protected:
            return
        rels=None
        tweets=None
        if user.followers_count>0 and user.friends_count>0:
            rels = self.twitter.get_edges(user._id)
            rels.attempt_save()

        if user.statuses_count>0:
            tweets = self.twitter.save_timeline(user._id,last_tid=settings.min_tweet_id)
        if tweets:
            user.next_crawl_date = datetime.utcnow()
            user.last_crawl_date = datetime.utcnow()
            user.tweets_per_hour = settings.tweets_per_hour
            user.last_tid = tweets[0]._id
        
        user.lookup_done = True
        if user.local_prob == 1.0 and not force:
            self.score_new_users(user, rels, tweets)

    def score_new_users(self, user, rels, tweets):
        jobs = defaultdict(LookupJobBody)
        jobs[user._id].done = True

        if rels:
            rfriends = rels.rfriends()
            if len(rfriends) < RFRIEND_POINTS:
                for u in rfriends:
                   jobs[u].rfriends_score = RFRIEND_POINTS/len(rfriends)

        if tweets:
            ats = defaultdict(int)
            for tweet in tweets:
                for uid in tweet.mentions:
                    ats[uid]+=1
            for u,c in ats.iteritems():
                points = c*MENTION_POINTS
                if points >0:
                    jobs[u].mention_score = points

        for k,j in jobs.iteritems():
            j._id = k
            j.put(self.stalk)


if __name__ == '__main__':
    if len(sys.argv) >1:
        if sys.argv[1]=='m':
            proc = LookupMaster()
        elif sys.argv[1]=='s':
            proc = LookupSlave('x')
        elif sys.argv[1]=='c':
            create_slaves(LookupSlave, prefix="x")
    else:
        print "spawning minions!"
        create_slaves(LookupSlave)
        proc = LookupMaster()
    proc.run()
