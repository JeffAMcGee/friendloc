from datetime import datetime

from base import models


class MockTwitterResource(object):
    def sleep_if_needed(self):
        #Calling sleep() in my unittests may be hazardous to your health!
        pass

    def get_edges(self, user_id):
        if user_id == 0:
            friends = followers = [1,3]
        elif user_id == 1:
            friends = followers = [x for x in xrange(10) if x!=1]
        else:
            friends = [1] + range(user_id*2,user_id*6,user_id)
            followers = [1] + [x for x in xrange(2,user_id) if not user_id%x]

        edges = models.Edges(_id=user_id,friends=friends,followers=followers)
        if user_id == 6:
            edges.friends.append(0)
            edges.followers.append(0)
        return edges

    def user_timeline(self, user_id):
        tweet = models.Tweet(
            _id = user_id,
            mentions = [],
            text = "howdy",
            created_at = datetime.utcnow(),
            user_id = user_id
        )

        if user_id==3:
            tweet.ats = [2]
        if user_id==6:
            tweet.ats = [7]
        return [tweet]*10


# FIXME: right now there is just a method that adds some immutable data to the
# database to run tests against. This needs some cleaning up.
def save_fixtures():
    save_users()

    twit = MockTwitterResource()
    for uid in xrange(10):
        twit.get_edges(uid).save()
        tweets = models.Tweets( _id=uid, tweets=twit.user_timeline(uid) )
        tweets.save()

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
    users[3].rfriends = [1]
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

