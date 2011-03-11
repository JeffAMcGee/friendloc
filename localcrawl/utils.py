import itertools

from settings import settings
from maroon import CouchDB, MongoDB


def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.izip_longest(*args, fillvalue=fillvalue)

def couch(name):
    return CouchDB(settings.couchdb_root+name,True)

def mongo(name):
    return MongoDB(name=name)


def in_local_box(place):
    box = settings.local_box
    return all(box[d][0]<place[d]<box[d][1] for d in ('lat','lng'))


def peek(iterable):
    it = iter(iterable)
    first = it.next()
    return first, itertools.chain([first],it)
