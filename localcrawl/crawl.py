#!/usr/bin/env python
import pdb
from datetime import datetime, timedelta
import logging

from maroon import *

from base.splitproc import SplitProcess
from base.models import User, Tweet
from base.twitter import TwitterResource
from settings import settings


class CrawlProcess(SplitProcess):
    def __init__(self,db_name,**kwargs):
        SplitProcess.__init__(self, **kwargs)
        self.db_name = db_name
        self.waiting = set()

    def produce(self):
        Model.database = MongoDB(name=self.db_name,host=settings.db_host)
        endtime = datetime.utcnow()
        return User.find(
                User.next_crawl_date<endtime,
                sort=User.next_crawl_date,
                timeout=False)

    def map(self,items):
        self.twitter = TwitterResource()
        Model.database = MongoDB(name=self.db_name,host=settings.db_host)

        for user in items:
            try:
                self.crawl(user)
                self.twitter.sleep_if_needed()
            except Exception as ex:
                logging.exception("exception for user %s"%user.to_d())
            yield None

    def crawl(self, user):
        logging.info("visiting %s - %s",user._id,user.screen_name)
        tweets = self.twitter.save_timeline(user._id, user.last_tid)
        if tweets:
            user.last_tid = tweets[0]._id
        logging.info("saved %d for %s",len(tweets),user.screen_name)
        now = datetime.utcnow()
        last = user.last_crawl_date if user.last_crawl_date is not None else datetime(2010,11,12)
        delta = now - last
        seconds = delta.seconds + delta.days*24*3600
        tph = (3600.0*len(tweets)/seconds + user.tweets_per_hour)/2
        user.tweets_per_hour = tph
        hours = min(settings.tweets_per_crawl/tph, settings.max_hours)
        user.next_crawl_date = now+timedelta(hours=hours)
        user.last_crawl_date = now
        user.save()

def crawl_debug(region):
    proc = CrawlProcess(region,
            label=region,
            slaves=1,
            log_level=logging.DEBUG)
    proc.run_single()
