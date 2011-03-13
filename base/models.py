from datetime import datetime
import json
from couchdbkit import ResourceConflict
import logging

import maroon
from maroon import *

from settings import settings

__all__ = [
    'TwitterModel',
    'TwitterIdProperty',
    'TwitterDateTimeProperty',
    'GeonamesPlace',
    'User',
    'Tweet',
    'Edges',
    'JobBody',
    'LookupJobBody',
]


class TwitterModel(Model):
    def __init__(self, from_dict=None, **kwargs):
        Model.__init__(self, from_dict, **kwargs)
        if self._id is None and from_dict and 'id' in from_dict:
            self._id = from_dict['id']
    
    def attempt_save(self):
        try:
            self.save()
        except ResourceConflict:
            logging.warn("conflict on %s %s",self.__class__.__name__,self._id)

class TwitterIdProperty(IntProperty):
    pass

class TwitterDateTimeProperty(DateTimeProperty):
    def  __init__(self, name, **kwargs):
        format="%a %b %d %H:%M:%S +0000 %Y"
        DateTimeProperty.__init__(self, name, format, **kwargs)

class GeonamesPlace(ModelPart):
    lat = FloatProperty('lat')
    lng = FloatProperty('lng')
    feature_code = TextProperty('code')
    name = TextProperty('name')
    population = IntProperty('pop')

class User(TwitterModel):
    _id = TwitterIdProperty('_id')

    #local properties
    tweets_per_hour = FloatProperty('tph')
    lookup_done = BoolProperty('ld')
    next_crawl_date = DateTimeProperty('ncd')
    last_tid = TwitterIdProperty('ltid')
    last_crawl_date = DateTimeProperty('lcd')
    rfriends_score = IntProperty('rfs')
    mention_score = IntProperty('ats')
    local_prob = FloatProperty('prob')
    geonames_place = ModelProperty('gnp',GeonamesPlace)
    median_loc = Property('mloc')
    rfriends = ListProperty('rfrds')
    just_friends = ListProperty('jfrds')
    just_followers = ListProperty('jfols')
    
    #properties from twitter
    verified = BoolProperty("ver")
    created_at = TwitterDateTimeProperty('ca')
    description = TextProperty('descr')
    favourites_count = IntProperty('favc')
    followers_count = IntProperty('folc')
    friends_count = IntProperty('frdc')
    geo_enabled = BoolProperty('geo')
    lang = TextProperty('lang')
    listed_count = IntProperty('lsc')
    location = TextProperty('loc')
    name = TextProperty('name')
    profile_image_url = TextProperty('img')
    protected = BoolProperty('prot')
    screen_name = TextProperty('sn')
    statuses_count = IntProperty('stc')
    url = TextProperty('url')
    utc_offset = IntProperty('utco')

    @classmethod
    def with_location():
        return User.database.User.find({'mloc':{'$exists':1}})

    @classmethod
    def next_crawl(cls,endtime = None):
        if endtime is None:
            endtime = datetime.utcnow()
        if settings.db == 'couch':
            endkey=endtime.timetuple()[0:6]
            users = cls.database.paged_view('user/next_crawl',endkey)
            return (user['id'] for user in users)
        else:
            users = User.database.User.find({'ncd':{'$lt':endtime}}, {'_id':1})
            return (user['_id'] for user in users)

class Tweet(TwitterModel):
    _id = TwitterIdProperty('_id')
    mentions = ListProperty('ats',int) #based on entities

    #properties from twitter
    coordinates = Property('coord')
    created_at = TwitterDateTimeProperty('ca')
    favorited = BoolProperty('fav')
    geo = Property('geo')
    in_reply_to_status_id = TwitterIdProperty('rtt')
    in_reply_to_user_id = TwitterIdProperty('rtu')
    place = Property('plc')
    text = TextProperty('tx')
    user_id = TwitterIdProperty('uid')

    def __init__(self, from_dict=None, **kwargs):
        TwitterModel.__init__(self, from_dict, **kwargs)
        if self.user_id is None and 'user' in from_dict:
            self.user_id = from_dict['user']['id']
        if self.mentions is None and 'entities' in from_dict:
            ats = from_dict['entities']['user_mentions']
            self.mentions = [at['id'] for at in ats ]

    @classmethod
    def by_date(cls,start=None,end=None):
        if settings.db == 'couch':
            startkey=start.timetuple()[0:6] if start else None
            endkey=end.timetuple()[0:6] if end else None
            tweets = cls.database.paged_view( 'tweet/date', include_docs=True, startkey=start, endkey=end)
            return (Tweet(from_dict=t['doc']) for t in tweets)
        else:
            q = {}
            if start: q['$gte']=start
            if end: q['$lt']=end
            tweets = Tweet.database.Tweet.find({'ca':q})
            return (Tweet(from_dict=t) for t in tweets)


class Edges(TwitterModel):
    # I only store the first 5000 friends and followers
    _id = TwitterIdProperty('_id')
    friends = ListProperty('frs',int)
    followers = ListProperty('fols',int)
    
    @classmethod
    def get_for_user_id(cls, _id):
        return cls.get_id(_id)

    def rfriends(self):
        #figure out whether the user has more friends or followers
        lil,big = sorted([self.friends,self.followers],key=len)
        return set(lil).intersection(big)

class JobBody(ModelPart):
    def put(self, stalk):
        stalk.put(json.dumps(self.to_d()),ttr=settings.beanstalkd_ttr)

    @classmethod
    def from_job(cls, job):
        return cls(json.loads(job.body))

class LookupJobBody(JobBody):
    _id = TwitterIdProperty('_id')
    rfriends_score = IntProperty('rfs')
    mention_score = IntProperty('ats')
    done = BoolProperty('done')
    force = BoolProperty('f')
