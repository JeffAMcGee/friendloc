import json
from itertools import chain

from maroon import *

from settings import settings

__all__ = [
    'TwitterModel',
    'TwitterIdProperty',
    'TwitterDateTimeProperty',
    'GeonamesPlace',
    'User',
    'Tweet',
    'Tweets',
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
        self.save()

class TwitterIdProperty(IntProperty):
    pass

class TwitterDateTimeProperty(DateTimeProperty):
    def  __init__(self, name, **kwargs):
        format="%a %b %d %H:%M:%S +0000 %Y"
        DateTimeProperty.__init__(self, name, format, **kwargs)

class GeonamesPlace(ModelPart):
    ignored = [
        'country_code','country_flag_url','country_name','elevation',
        'feature_class','fully_qualified_name','google_map_url','gtopo30',
        'name_ascii','placetype','score','timezone','yahoo_map_url','zipcode',
    ]
    feature_id = IntProperty('fid')
    lat = FloatProperty('lat')
    lng = FloatProperty('lng')
    feature_code = TextProperty('code')
    name = TextProperty('name')
    mdist = FloatProperty('mdist')

    def to_tup(self):
        return (self.lng,self.lat)


class User(TwitterModel):
    _id = TwitterIdProperty('_id')
    ignored = [
        'contributors_enabled', 'follow_request_sent', 'following',
        'profile_background_color', 'profile_background_image_url',
        'profile_background_tile', 'profile_link_color',
        'profile_sidebar_border_color', 'profile_sidebar_fill_color',
        'profile_text_color', 'profile_use_background_image',
        'show_all_inline_media', 'time_zone', 'status', 'notifications',
        'default_profile', 'default_profile_image', 'profile_image_url_https',
        'profile_background_image_url_https',
        'id', 'id_str', 'is_translator'
    ]
    NEBR_KEYS = ['rfriends','just_followers','just_friends','just_mentioned']

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
    median_loc = Property('mloc') # longitude, latitude tuple
    pred_loc = Property('ploc') # longitude, latitude tuple
    rfriends = ListProperty('rfrds')
    just_friends = ListProperty('jfrds')
    just_followers = ListProperty('jfols')
    just_mentioned = ListProperty('jats')
    neighbors = ListProperty('nebrs')
    many_edges = BoolProperty('mne')
    mod_group = IntProperty('mdg')
    error_status = IntProperty('err')
    local_friends = FloatProperty('lofrd')
    local_followers = FloatProperty('lofol')
    local_ratio = FloatProperty('lorat')
    crowd_id = IntProperty('cid')

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

    def __init__(self, from_dict=None, **kwargs):
        TwitterModel.__init__(self, from_dict, **kwargs)
        if self.median_loc and self._id is not None:
            self.mod_group = self._id%100

    @classmethod
    def with_location(cls):
        return User.database.User.find({'mloc':{'$exists':1}})

    @classmethod
    def mod_id(cls, user_or_id):
        if isinstance(user_or_id, User):
            id = user._id
        elif isinstance(user_or_id, dict):
            id = user_or_id['id']
        else:
            id = user_or_id
        return "%02d"%(id%100)

    @classmethod
    def find_connected(cls,**kwargs):
        return User.find(
            User.just_friends.exists() & User.just_followers.exists() & User.rfriends.exists(),
            **kwargs
            )

    def has_place(self):
        gnp = self.geonames_place
        return gnp and gnp.mdist<1000

    @property
    def contacts(self):
        return tuple(chain.from_iterable(
                    getattr(self,key) or ()
                    for key in self.NEBR_KEYS
                    ))

    @classmethod
    def mod_id_set(cls, group):
        coll = cls.database.User
        query = {'_id':{'$mod':[100,group]}}
        return {u['_id'] for u in coll.find(query,fields=[])}


class Tweet(TwitterModel):
    _id = TwitterIdProperty('_id')
    mentions = ListProperty('ats',int) #based on entities

    ignored = [
        'contributors', 'entities', 'in_reply_to_screen_name', 'source',
        'truncated', 'user', 'id', 'id_str', 'retweeted', 'retweeted_status',
        'retweeted_count', 'retweet_count', 'favorited', 'geo', 'user_id_str',
        'possibly_sensitive_editable', 'possibly_sensitive',
        ]

    #properties from twitter
    coordinates = Property('coord')
    created_at = TwitterDateTimeProperty('ca')
    in_reply_to_status_id = TwitterIdProperty('rtt')
    in_reply_to_user_id = TwitterIdProperty('rtu')
    place = Property('plc')
    text = TextProperty('tx')
    user_id = TwitterIdProperty('uid')

    #properties from us
    crowd_id = IntProperty('cid')

    def __init__(self, from_dict=None, **kwargs):
        TwitterModel.__init__(self, from_dict, **kwargs)
        if self.user_id is None and 'user' in from_dict:
            self.user_id = from_dict['user']['id']
        if self.mentions is None and 'entities' in from_dict:
            ats = from_dict['entities']['user_mentions']
            self.mentions = [at['id'] for at in ats ]

    @classmethod
    def by_date(cls,start=None,end=None):
        q = {}
        if start: q['$gte']=start
        if end: q['$lt']=end
        tweets = Tweet.database.Tweet.find({'ca':q})
        return (Tweet(from_dict=t) for t in tweets)


class Tweets(TwitterModel):
    def __init__(self, from_dict=None, **kwargs):
        TwitterModel.__init__(self, from_dict, **kwargs)
        if self.tweets and not self.ats:
            ats = set(at for t in self.tweets for at in t.mentions)
            ats.discard(self._id)
            self.ats = ats

    _id = TwitterIdProperty('_id') #user id
    ats = ListProperty('ats',int)
    tweets = ModelListProperty('tws',Tweet)


class Edges(TwitterModel):
    # I only store the first 5000 friends and followers
    _id = TwitterIdProperty('_id') #user id
    friends = ListProperty('frs',int)
    followers = ListProperty('fols',int)
    #This is only used if many_edges is true for the user
    lookups = ListProperty('lkus',int)
    
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


class Crowd(TwitterModel):
    _id = IntProperty('_id')
    loc = Property('mloc') # longitude, latitude tuple
    uids = ListProperty('uids')
    edges = ListProperty('edges')
    # consider other fields from crowdy


class Cluster(ModelPart):
    loc = Property('mloc') # longitude, latitude tuple
    cids = ListProperty('cids')
    size = IntProperty('size')


class Topic(TwitterModel):
    _id = TextProperty('_id')
    clusters = ModelListProperty('cls',Cluster)

