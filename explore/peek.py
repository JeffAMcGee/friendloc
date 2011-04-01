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
from base.gisgraphy import GisgraphyResource
from base.models import *
from base.utils import *


def find_geo_ats(path=None):
    uids = defaultdict(set)
    for t in read_json(path):
        ats = t['ats']
        if ats:
            uids[t['uid']].update(ats)
    for k,v in uids.iteritems():
        print json.dumps(dict(uid=k,ats=list(v)))


def find_removed_locs():
    for user in User.find(User.median_loc.exists()):
        for key in ('just_friends','just_followers','rfriends'):
            l = getattr(user,key)
            if not l: continue
            amigo = User.get_id(l[0])
            place = amigo.geonames_place.to_d()
            if not place:
                print amigo._id

def fix_removed_locs(path=None):
    lusers = set(read_json ('removed_locs'))
    for user in read_json(path):
        id=user['_id']
        if id in lusers and user['gnp']:
            obj = User.get_id(user['_id'])
            obj.geonames_place = GeonamesPlace(from_dict=user['gnp'])
            obj.save()
            lusers.remove(id)
    print len(lusers)
 
def save_geo_mloc(path=None):
    gis = GisgraphyResource()
    for d in read_json(path):
        if 'mloc' in d:
            user = User(from_dict=d)
            seen.add(user._id)
            user.geonames_place = gis.twitter_loc(user.location)
            user.save()

def save_amigos(path=None):
    users = set()
    amigos = set()
    for user in User.get_all(fields=['jfrds','jfols','rfrds']):
        users.add(user._id)
        for l in (user.just_friends,user.just_followers,user.rfriends):
            if l:
                amigos.add(l[0])
    missing = amigos-users
    print "missing %d"%len(missing)
    for d in read_json(path):
        if d['_id'] in missing:
            user = User(from_dict=d)
            user.save()
            missing.remove(user._id)
    print "missing %d"%len(missing)


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
    edges = ModelCache(Edges)
    data = []
    for user in User.find( User.rfriends.exists() ):
        amigo_id = user.rfriends[0]
        amigo = User.get_id(amigo_id, fields=['gnp'])
        me = edges[user._id]
        you = edges[amigo_id]
        sets = dict(
            mfrd = set(me.friends),
            mfol = set(me.followers),
            yfrd = set(you.friends),
            yfol = set(you.followers),
            )
        all = (sets['mfrd']|sets['mfol'])&(sets['yfrd']|sets['yfol'])
        dist = coord_in_miles(user.median_loc,amigo.geonames_place.to_d())
        d = dict(dist=dist, all=len(all))
        for k,v in sets.iteritems():
            d['l'+k]= len(v)
            d[k] = list(all&v)
        data.append(d)
    data.sort(key=itemgetter('dist'))
    for d in data:
        print json.dumps(d)
