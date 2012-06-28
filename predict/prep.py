import random

from base.utils import grouper
from base import gob
from base.models import User, Tweets, Edges


@gob.mapper(all_items=True)
def training_users(uids):
    for g in grouper(100,uids,dontfill=True):
        ids_group = tuple(g)
        if ids_group[0]%100<50:
            for u in User.find(User._id.is_in(ids_group)):
                yield u.to_d()


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
    return (res,) # mappers return an iterable


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
