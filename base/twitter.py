from restkit import OAuthFilter, Resource
import restkit.oauth2 as oauth
import json
import time
import logging
from datetime import datetime
from restkit.errors import RequestFailed, Unauthorized
from http_parser.http import NoMoreData
from settings import settings
from models import Edges, User, Tweet


class TwitterResource(Resource):
    # When a request fails, we retry with an exponential backoff from
    # 15 to 240 seconds.
    backoff_seconds = [15,60,240,0]

    def __init__(self):
        consumer = oauth.Consumer(
                key=settings.consumer_key,
                secret=settings.consumer_secret)
        token = oauth.Token(
                key=settings.token_key,
                secret=settings.token_secret)
        url = "http://api.twitter.com/1/"
        auth = OAuthFilter('*', consumer, token)
        Resource.__init__(self,
                url,
                filters=[auth],
                client_opts={'timeout':60}
        )
        self.remaining = 10000

    def get_d(self, path=None, headers=None, **kwargs):
        """
        GET json from the twitter API and return it as a dict.

        If things fail, it will retry 4 times with an exponential backoff.
        If that doesn't work, it raises all sorts of stuff:
            404: ResourceNotFound
            401, 403: Unauthorized
            other errors: RequestFailed
        """
        for delay in self.backoff_seconds:
            try:
                r = self.get(path, headers, **kwargs)
                self._parse_ratelimit(r)
                return json.loads(r.body_string())
            except (ValueError,NoMoreData) as e:
                logging.error("incomplete response (%s)",type(e).__name__)
                if delay==0:
                    raise
            except Unauthorized as unauth:
                self._parse_ratelimit(unauth.response)
                raise
            except RequestFailed as failure:
                self._parse_ratelimit(failure.response)
                if failure.response.status_int == 502:
                    logging.info("Fail whale says slow down!")
                else:
                    logging.error("%s while retrieving %s",
                        failure.response.status,
                        failure.response.final_url
                    )
                if delay==0:
                    raise
                # FIXME: what should I do for 503?
                if failure.response.status_int in (400,420,503):
                    # The whale says slow WAY down!
                    delay = 240
            time.sleep(delay)

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
                "users/lookup.json",
                screen_name=names,
                user_id=ids,
                **kwargs
            )
        except RequestFailed:
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
        return self.get_ids("friends/ids.json", user_id)

    def followers_ids(self, user_id):
        return self.get_ids("followers/ids.json", user_id)

    def get_edges(self, user_id):
        return Edges(
                _id=user_id,
                friends=self.friends_ids(user_id),
                followers=self.followers_ids(user_id),
        )

    def user_timeline(self, user_id, count=100, **kwargs):
        timeline = self.get_d(
            "statuses/user_timeline.json",
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
            except Unauthorized:
                logging.warn("unauthorized!")
                break
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
        #sleep an extra minute in case clocks are wrong
        time.sleep(60+delta.seconds if delta.days==0 else 120)
