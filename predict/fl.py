import math

from base import gob
from base.utils import coord_in_miles

def logify(x):
    return int(math.ceil(math.log(x+1,2)))

@gob.mapper()
def edge_vect(user):
    for rel in user['rels']:
        flags = [rel['kind'] >>i & 1 for i in range(4)]
        vals = [logify(rel[k]) for k in ('mdist','folc')]
        dist = logify(coord_in_miles(user['mloc'],rel))
        yield flags+vals+[dist]
