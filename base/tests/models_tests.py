import unittest
from datetime import datetime

from base import models


# FIXME: right now there is just a method that adds some immutable data to the
# database to run tests against. This needs some cleaning up.


def save_fixtures():
    save_users()
    save_edges()
    save_tweets()


def save_users():
    names = [
        'Joyce','Alberto','Beryl','Chris','Debby','Ernesto','Florence',
        'Gordon','Helene','Issac',
        ]

    def cstx(index):
        # shift the city around a bit so we know who is where
        return models.GeonamesPlace(
            feature_id=4682464,
            lat=30+index*.1,
            lng=-96.334,
            mdist=1.890,
            name="College Station",
            )

    users = [
            models.User(
                _id = count,
                verified = True,
                followers_count = count*count,
                friends_count = 2+count*count,
                name = name,
                loc = "College Station, TX",
                screen_name = "{}_{}".format(name[0],count),
                gnp = cstx(count),
                mod_group = count,
                )
            for count, name in enumerate(names)
            ]
    users[3].median_loc = [-96,30]
    users[3].rfriends = [0]
    users[3].just_friends = [6,9]
    users[3].just_followers = []
    users[3].just_mentioned = [2]

    users[6].median_loc = [-96,31]
    users[6].rfriends = [0,1]
    users[6].just_friends = []
    users[6].just_followers = [2,3]
    users[6].just_mentioned = [7]

    users[7].protected = True
    for user in users:
        user.save()


def save_edges():
    models.Edges( _id=0, friends=[1,3], followers=[1,3] ).save()
    not_one = [x for x in xrange(10) if x!=1]
    models.Edges( _id=1, friends=not_one, followers=not_one ).save()
    for index in xrange(2,10):
        # users are followed by their factors
        edges = models.Edges(
            _id = index,
            friends = [1] + range(index*2,index*6,index),
            followers = [1] + [x for x in xrange(2,index) if not index%x],
            )
        edges.save()


def save_tweets():
    def tweet(index):
        return models.Tweet(
            _id = index,
            mentions = [],
            text = "howdy",
            created_at = datetime.utcnow(),
            user_id = index
        )
    for index in xrange(10):
        tweets = models.Tweets(
            _id = index,
            tweets = [tweet(index)]*10,
            ats = [],
            )
        if index==3:
            tweets.ats = [2]
        if index==6:
            tweets.ats = [7]
        tweets.save()


class TestSimpleEnv(unittest.TestCase):
    def setUp(self):
        pass

    def test_split_saver(self):
        self.assertEqual(1,1)

