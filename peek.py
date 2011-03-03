import itertools
import time
import os
import sys
import random
import logging
from collections import defaultdict
from datetime import datetime as dt
from operator import itemgetter

from settings import settings
import localcrawl.twitter as twitter
from localcrawl.models import *
from maroon import ModelCache


def all_users():
    return User.get_all()


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
         

def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.izip_longest(*args, fillvalue=fillvalue)


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


def krishna_export(start=[2010],end=None):
    "export the tweets for Krishna's crawler"
    view = Model.database.paged_view(
            'tweet/date',
            include_docs=True,
            startkey=start,
            endkey=end
        )
    for k,g in itertools.groupby(view,itemgetter('key')):
        path = os.path.join(*(str(x) for x in k))
        mkdir_p(os.path.dirname(path))
        with open(path,'w') as f:
            for t in (row['doc'] for row in g):
                ts = int(time.mktime(dt(*t['ca']).timetuple()))
                if t['ats']:
                    for at in t['ats']:
                        print>>f,"%d %s %s %s"%(ts,t['_id'],t['uid'],at)
                else:
                    print>>f,"%d %s %s"%(ts,t['_id'],t['uid'])


def _tri_users_dict_set(users_path):
    users = dict((int(d['id']),d) for d in _read_json(users_path))
    return users,set(users)


def find_ats(users_path="hou_tri_users"):
    users, uids = _tri_users_dict_set(users_path)
    for line in sys.stdin:
        d = json.loads(line)
        if d.get('ats'):
            local = int(d['uid']) in uids
            for at in d['ats']:
                if local or int(at) in uids:
                    print "%s\t%s"%(d['uid'],at)


def print_tri_counts(users_path="hou_tri_users"):
    users, uids = _tri_users_dict_set(users_path)
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
            dist = _coord_in_miles(user,users[your_id]),
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
