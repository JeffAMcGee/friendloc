import random
import bisect
import itertools
from itertools import chain

from base.utils import grouper
from base import gob
from base.models import User, Tweets, Edges


NEBR_FLAGS = {
    'fols':4,
    'frds':2,
    'ated':1,
}


@gob.mapper(all_items=True)
def training_users(uids):
    for g in grouper(100,uids,dontfill=True):
        ids_group = tuple(g)
        if ids_group[0]%100<50:
            for u in User.find(User._id.is_in(ids_group)):
                yield u.to_d()


class NeighborsDict(object):
    def __init__(self,env):
        self.env = env
        mloc_blur = env.load('mloc_blur','mp')
        self.mb_omit = next(mloc_blur)
        self.mb_buckets = next(mloc_blur)
        self.mb_ratios = next(mloc_blur)

    def _prep_nebr(self,nebr):
        kind = sum(
                bit if nebr._id in self.contacts[key] else 0
                for key,bit in NEBR_FLAGS.iteritems()
                )
        return dict(
            folc=nebr.friends_count,
            frdc=nebr.followers_count,
            lofrd=nebr.local_friends,
            lofol=nebr.local_followers,
            lat=nebr.geonames_place.lat,
            lng=nebr.geonames_place.lng,
            mdist=nebr.geonames_place.mdist,
            kind=kind,
            prot=nebr.protected,
            _id=nebr._id,
            )

    def _blur_gnp(self, user_d):
        gnp = user_d.get('gnp')
        if not gnp or gnp['mdist']>1000:
            return None
        if random.random()>self.mb_omit:
            return None
        index = bisect.bisect(self.mb_buckets,gnp['mdist'])
        ratio = self.mb_ratios[index]
        # exaggerate the error that is already there according to the ratio
        for key,real in zip(('lng','lat'),user_d['mloc']):
            delta = real-gnp[key]
            gnp[key] = real+ratio*delta
        gnp['mdist'] = ratio*gnp['mdist']
        return gnp

    @gob.mapper()
    def nebrs_d(self,user_d):
        nebrs = User.find(User._id.is_in(user_d['nebrs']))
        tweets = Tweets.get_id(user_d['_id'],fields=['ats'])
        rfrds = set(user_d['rfrds'])

        self.contacts = dict(
            ated = set(tweets.ats or []),
            frds = rfrds.union(user_d['jfrds']),
            fols = rfrds.union(user_d['jfols']),
        )

        res = dict(
            _id = user_d['_id'],
            mloc = user_d['mloc'],
            nebrs = map(self._prep_nebr,nebrs),
            gnp = self._blur_gnp(user_d),
            )
        yield res


@gob.mapper()
def edge_d(user):
    # FIXME: I think this function is no longer useful.
    tweets = Tweets.get_id(user['_id'],fields=['ats'])
    edges = Edges.get_id(user['_id'])
    ats = set(tweets.ats or [])
    frds = set(edges.friends or [])
    fols = set(edges.followers or [])

    group_order = (ats,frds,fols)
    def _group(uid):
        #return 7 for ated rfrd, 4 for ignored jfol
        return sum(2**i for i,s in enumerate(group_order) if uid in s)

    #pick the 100 best users
    lookups = edges.lookups if edges.lookups else list(ats|frds|fols)
    random.shuffle(lookups)
    lookups.sort(key=_group, reverse=True)
    lookups = lookups[:400]

    #get the users - this will be SLOW
    amigos = User.find(
            User._id.is_in(lookups) & User.geonames_place.exists(),
            fields =['gnp','folc','prot'],
            )

    rels = [
        _rel_d(amigo, _group(amigo._id) + (8 if amigo.protected else 0))
        for amigo in amigos]
    rels.sort(key=lambda d: d['kind']%8, reverse=True)
    res = dict(
        _id = user['_id'],
        mloc = user['mloc'],
        lu_len = len(lookups),
        rels = rels,
        )
    if user.get('gnp'):
        res['gnp'] = user['gnp']
    yield res


def _rel_d(user, kind):
    gnp = user.geonames_place.to_d()
    return dict(
        folc=user.followers_count,
        lat=gnp['lat'],
        lng=gnp['lng'],
        mdist=gnp['mdist'],
        kind=kind,
        _id=user._id,
        )



class MlocBlur(object):
    def __init__(self,env):
        self.env = env

    @gob.mapper()
    def mloc_blur(self):
        cutoff = 250000
        mdists = {}
        for key in ('mloc','contact'):
            files = self.env.split_files(key+'_mdist')
            items_ = chain.from_iterable(self.env.load(f,'mp') for f in files)
            mdists[key] = filter(None,itertools.islice(items_,cutoff))
        yield 1.0*len(mdists['contact'])/len(mdists['mloc'])

        count = len(mdists['contact'])
        step = count//100
        mdists['mloc'] = mdists['mloc'][:count]
        for key,items in mdists.iteritems():
            mdists[key] = sorted(items)
        # the boundaries of the 100 buckets
        yield mdists['mloc'][step:step*100:step]

        ml_pts = np.array(mdists['mloc'][step/2::step])
        ct_pts = np.array(mdists['contact'][step/2::step])
        # the ratio at the middle of the buckets
        yield list(ct_pts/ml_pts)
