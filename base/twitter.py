import json
import time
import logging
from datetime import datetime

from oauth_hook import OAuthHook
import requests

from settings import settings
from models import Edges, User, Tweet


class TwitterFailure(Exception):
    def __init__(self,response):
        msg = "%d for %r"%(response.status_code,response.request)
        super(TwitterFailure,self).__init__(msg)
        self.response = response
        self.status_code = response.status_code


class TwitterResource(object):
    # When a request fails, we retry with an exponential backoff from
    # 15 to 240 seconds.
    backoff_seconds = [20,60,180,540,0]

    def __init__(self):
        oauth_hook = OAuthHook(
                        settings.token_key,
                        settings.token_secret,
                        settings.consumer_key,
                        settings.consumer_secret,
                        header_auth=True)
        self.session = requests.session(
                        hooks={'pre_request': oauth_hook},
                        timeout=60,
                        config=dict(max_retries=2,safe_mode=True),
                        prefetch=True,
                        )
        self.remaining = 10000

    def get_d(self, path=None, **kwargs):
        """
        GET json from the twitter API and return it as a dict.

        If things fail, it will retry 5 times with an exponential backoff.
        If that doesn't work, it raises a TwitterFailure.
        """
        for delay in self.backoff_seconds:
            url = "http://api.twitter.com/1/%s.json"%path
            resp = requests.get(url, params=kwargs, session=self.session)
            self._parse_ratelimit(resp)

            if resp.status_code == 200:
                return json.loads(resp.text)
            elif resp.status_code in (401,403,404):
                raise TwitterFailure(resp)
            elif resp.status_code == 502:
                logging.info("Fail whale says slow down!")
            else:
                logging.error("%s while retrieving %s",
                    resp.status_code,
                    resp.url,
                )
                if resp.status_code == 0:
                    logging.error("error: %r",resp.error)
                if resp.status_code in (400,420,503):
                    # The whale says slow WAY down!
                    delay = 240
            time.sleep(delay)
        raise TwitterFailure(resp)

    def _parse_ratelimit(self,r):
        if 'X-RateLimit-Remaining' in r.headers:
            self.remaining = int(r.headers['X-RateLimit-Remaining'])
            if self.remaining%500==0:
                logging.info("api calls remaining: %d",self.remaining)
            stamp = int(r.headers['X-RateLimit-Reset'])
            self.reset_time = datetime.utcfromtimestamp(stamp)
            if self.remaining < 25:
                self.sleep_if_needed()


    def get_ids(self, path, user_id, **kwargs):
        ids=self.get_d(
            path=path,
            user_id=user_id,
            cursor=-1,
            **kwargs
        )
        return ids['ids']

    def user_lookup(self, user_ids=[], screen_names=[], **kwargs):
        ids = ','.join(str(u) for u in user_ids)
        names = ','.join(screen_names)
        try:
            lookup = self.get_d(
                "users/lookup",
                screen_name=names,
                user_id=ids,
                **kwargs
            )
        except TwitterFailure:
            #ick. Twitter dies for some users.  Do a binary search to avoid them.
            if len(user_ids)>1:
                split = len(user_ids)/2
                first,last = user_ids[:split],user_ids[split:]
                logging.info("split to %r and %r",first,last)
                return self.user_lookup(first) + self.user_lookup(last)
            elif user_ids:
                logging.warn("Twitter hates %d",user_ids[0])
                return [None]
        users = [User(d) for d in lookup]
        if len(users)==len(user_ids) or screen_names:
            return users
        # Ick. Twitter just removes suspended users from the results.
        d = dict((u._id,u) for u in users)
        return [d.get(uid,None) for uid in user_ids]

    def friends_ids(self, user_id):
        return self.get_ids("friends/ids", user_id)

    def followers_ids(self, user_id):
        return self.get_ids("followers/ids", user_id)

    def get_edges(self, user_id):
        return Edges(
                _id=user_id,
                friends=self.friends_ids(user_id),
                followers=self.followers_ids(user_id),
        )

    def user_timeline(self, user_id, count=100, **kwargs):
        timeline = self.get_d(
            "statuses/user_timeline",
            user_id=user_id,
            trim_user=1,
            include_rts=1,
            include_entities=1,
            count=count,
            **kwargs
        )
        return [Tweet(t) for t in timeline]

    def save_timeline(self, uid, last_tid, max_tid=None):
        since_id = int(last_tid)-1
        max_id = int(max_tid)-1 if max_tid else None

        all_tweets = []
        while since_id != max_id:
            try:
                tweets = self.user_timeline(
                    uid,
                    max_id = max_id,
                    since_id = since_id,
                    count=160,
                )
            except TwitterFailure as fail:
                if fail.status_code in (401,403):
                    logging.warn("unauthorized!")
                    break
                raise
            if not tweets:
                logging.warn("no tweets after %d for %s",len(all_tweets),uid)
                break
            all_tweets+=tweets
            if len(tweets)<140:
                #there are no more tweets, and since_id+1 was deleted
                break
            max_id =int(tweets[-1]._id)-1
            if len(all_tweets)>=3150:
                logging.error("hit max after %d for %s",len(all_tweets),uid)
                break
        stored_tweets = [t for t in all_tweets if int(t._id)-1>since_id]
        Tweet.database.bulk_save_models(stored_tweets)
        return all_tweets

    def sleep_if_needed(self):
        logging.info("api calls remaining: %d",self.remaining)
        if self.remaining > 100: return
        delta = (self.reset_time-datetime.utcnow())
        logging.info("goodnight for %r",delta)
        if delta.days==0:
            #sleep an extra minute in case clocks are wrong
            time.sleep(max(60+delta.seconds,3600))
        else:
            time.sleep(120)
