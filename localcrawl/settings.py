class SettingsBunch(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

settings = SettingsBunch(
    # FIXME: rename region to mongo_db; it is just the database name now
    region = "friendloc",
    db_host = "localhost",
    gisgraphy_url = "http://services.gisgraphy.com",
    mongo_host = 'localhost',
    log_dir = 'logs',
    log_crashes = True,

    #just for localcrawl
    slaves = 8,
    utc_offset = -21600,
    non_local_cutoff = 13,
    min_cutoff = 2,
    lookup_in = 'lookup.in',
    lookup_out = 'lookup.out',
    tweets_per_hour = .04,  # 1 tweet/day is median
    tweets_per_crawl = 200,
    max_hours = 360,  # 15 days
    min_tweet_id = 25000000000000000,
    beanstalk_host = 'localhost',
    beanstalk_port = 11300,
    beanstalkd_ttr = 7200,
    mention_weight = .5,
    crawl_ratio = .1,
    local_box = dict(lat=(29,30.5),lng=(-96,-94.5)),
)

try:
    from settings_dev import settings as s
    settings.update(s)
except ImportError:
    pass
