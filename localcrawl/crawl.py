#!/usr/bin/env python
from collections import defaultdict
import beanstalkc
import json
import pdb
import signal
from datetime import datetime, timedelta
import time
from itertools import groupby
from restkit import Unauthorized
import logging
import multiprocessing
import Queue

import maroon
from maroon import *

from base.models import User, Tweet
from base.twitter import TwitterResource
from settings import settings
from base.gisgraphy import GisgraphyResource
from procs import LocalProc, create_slaves


HALT = False
def set_halt(x=None,y=None):
    print "halting"
    global HALT
    HALT=True
#signal.signal(signal.SIGINT, set_halt)
#signal.signal(signal.SIGUSR1, set_halt)


class CrawlMaster(LocalProc):
    def __init__(self):
        LocalProc.__init__(self,"crawl")
        self.waiting = set()
        self.todo = multiprocessing.JoinableQueue(30)
        self.done = multiprocessing.Queue()

    def run(self):
        print "started crawl"
        logging.info("started crawl")
        try:
            while not HALT:
                self.queue_crawl()
                print "naptime"
                time.sleep(600)
        except:
            logging.exception("exception caused HALT")
        print "done"

    def queue_crawl(self):
        logging.info("queue_crawl")
        for uid in User.next_crawl():
            if HALT: break
            if uid in self.waiting: continue # they are queued
            self.waiting.add(uid)
            self.todo.put(uid)

            if len(self.waiting)%100==0:
                # let the queue empty a bit
                self.read_crawled()

    def read_crawled(self):
        logging.info("read_crawled, %d",len(self.waiting))
        try:
            while True:
                uid = self.done.get_nowait()
                self.waiting.remove(uid)
        except Queue.Empty:
            return


class CrawlSlave(LocalProc):
    def __init__(self, slave_id, todo, done):
        LocalProc.__init__(self,'crawl', slave_id)
        self.twitter = TwitterResource()
        self.todo = todo
        self.done = done

    def run(self):
        #pdb.Pdb(stdin=open('/dev/stdin', 'r+'), stdout=open('/dev/stdout', 'r+')).set_trace()
        while not HALT:
            user=None
            try:
                uid = self.todo.get()
                user = User.get_id(uid)
                self.crawl(user)
                self.done.put(uid)
                self.todo.task_done()
                if self.twitter.remaining < 10:
                    dt = (self.twitter.reset_time-datetime.utcnow())
                    logging.info("goodnight for %r",dt)
                    time.sleep(dt.seconds)
            except Exception as ex:
                if user:
                    logging.exception("exception for user %s"%user.to_d())
                else:
                    logging.exception("exception and user is None")
            logging.info("api calls remaining: %d",self.twitter.remaining)
        print "slave is done"

    def crawl(self, user):
        logging.debug("visiting %s - %s",user._id,user.screen_name)
        tweets = self.twitter.save_timeline(user._id, user.last_tid)
        if tweets:
            user.last_tid = tweets[0]._id
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


if __name__ == '__main__':
    proc = CrawlMaster()
    create_slaves(CrawlSlave, proc.todo, proc.done)
    proc.run()
