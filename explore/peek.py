import itertools
import time
import os
import sys
import random
import logging
from collections import defaultdict
from datetime import datetime as dt
from operator import itemgetter

from maroon import ModelCache

from settings import settings
import base.twitter as twitter
from base.models import *
from base.utils import *


def print_locs(start='T',end='U'):
    view = Model.database.paged_view('tweet/plc',
            include_docs=True,
            startkey=None,
            endkey=None,
            startkey_docid=start,
            endkey_docid=end,
            )
    for row in view:
        if 'coord' in row['doc']:
            c = row['doc']['coord']['coordinates']
            print '%f\t%f\t%s'%(c[0],c[1],row['doc']['uid'])


def count_recent():
    min_int_id = 8000000000000000L
    view = Model.database.paged_view('user/and_tweets')
    for k,g in itertools.groupby(view, lambda r:r['key'][0]):
            user_d = g.next()
            if user_d['id'][0] != 'U':
                print "fail %r"%user_d
                continue
            tweets = sum(1 for r in g if as_int_id(r['id'])>min_int_id)
            print "%d\t%s"%(tweets,user_d['id'])
         


def count_sn(path):
    "used to evaluate the results of localcrawl"
    lost =0
    found =0
    sns = (sn.strip() for sn in open(path))
    for group in grouper(100,sns):
        for user in res.user_lookup([], screen_names=group):
            if user._id in Model.database:
                found+=1
                print "found %s - %s"%(user.screen_name,user._id)
            else:
                lost+=1
                print "missed %s - %s"%(user.screen_name,user._id)
    print "lost:%d found:%d"%(lost,found)


def _tweets_correct_date(start,end):
    tweets = Tweet.find(
            Tweet._id.range(start and int(start), end and int(end)),
            sort='_id',
            descending = True,
            timeout=False)
    oldest = dt.utcnow()
    for tweet in tweets:
        if (tweet.created_at-oldest).days>0:
            print "tweet out of order - %r"%tweet.to_d()
        oldest = tweet.created_at = min(oldest,tweet.created_at)
        yield tweet
 
def krishna_export(start=settings.min_tweet_id,end=None):
    "export the tweets for Krishna's crawler"
    tweets = _tweets_correct_date(start,end)
    for day,group in itertools.groupby(tweets,lambda t: t.created_at.date()):
        path = os.path.join(*(str(x) for x in day.timetuple()[0:3]))
        mkdir_p(os.path.dirname(path))
        print path
        with open(path,'w') as f:
            for t in group:
                ts = int(time.mktime(t.created_at.timetuple()))
                if t.mentions:
                    for at in t.mentions:
                        print>>f,"%d %d %d %d"%(ts,t._id,t.user_id,at)
                #else:
                #    print>>f,"%d %d %d"%(ts,t._id,t.user_id)


def find_ats(users_path="hou_tri_users"):
    users, uids = tri_users_dict_set(users_path)
    for line in sys.stdin:
        d = json.loads(line)
        if d.get('ats'):
            local = int(d['uid']) in uids
            for at in d['ats']:
                if local or int(at) in uids:
                    print "%s\t%s"%(d['uid'],at)


def print_tri_counts(users_path="hou_tri_users"):
    users, uids = tri_users_dict_set(users_path)
    edges = ModelCache(Edges)
    data = []
    for uid,user in users.iteritems():
        me = edges[uid]
        if not me : continue
        friends = uids.intersection(me.friends)
        if not friends: continue
        your_id = random.sample(friends,1)[0]
        you = edges[your_id]
        sets = dict(
            mfrd = set(me.friends),
            mfol = set(me.followers),
            yfrd = set(you.friends),
            yfol = set(you.followers),
            )
        all = (sets['mfrd']|sets['mfol'])&(sets['yfrd']|sets['yfol'])
        d = dict(
            dist = coord_in_miles(user,users[your_id]),
            all = len(all),
            rfriend = 1 if your_id in sets['mfol'] else 0 
        )
        for k,v in sets.iteritems():
            d['l'+k]= len(v)
            d[k] = list(all&v)
        data.append(d)
    data.sort(key=itemgetter('dist'))
    for d in data:
        print json.dumps(d)
