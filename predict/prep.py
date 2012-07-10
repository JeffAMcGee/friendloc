import random

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

@gob.mapper()
def nebrs_d(user_d):
    nebrs_ = User.find(User._id.is_in(user_d['nebrs']))
    tweets = Tweets.get_id(user_d['_id'],fields=['ats'])
    rfrds = set(user_d['rfrds'])

    contacts = dict(
        ated = set(tweets.ats or []),
        frds = rfrds.union(user_d['jfrds']),
        fols = rfrds.union(user_d['jfols']),
    )

    nebrs = []
    for nebr in nebrs_:
        kind = sum(
                bit if nebr._id in contacts[key] else 0
                for key,bit in NEBR_FLAGS.iteritems()
                )
        nebrs.append(dict(
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
            ))
    res = dict(
        _id = user_d['_id'],
        mloc = user_d['mloc'],
        nebrs = nebrs,
        )
    if user_d.get('gnp'):
        res['gnp'] = user_d['gnp']
    yield res


@gob.mapper()
def edge_d(user):
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
