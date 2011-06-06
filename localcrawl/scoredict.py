from collections import defaultdict
from math import log

from settings import settings


# This module exists because my machine can handle a dict of 100Mil ints,
# but not tuples.  Sticking this in the db will make things orders of
# magnitude slower.
#
# The scores dict contains int bitfields:
# bit 0-13: mention_score
# bit 14-27: rfriends_score
# bit 28-29: state - NEW, LOOKUP, DONE, FAILED

BUCKETS = 29
MAX_SCORE = 1<<14
STATE_FIELD = MAX_SCORE*MAX_SCORE
NEW=0
LOOKUP=1
DONE=2
FAILED=3


def log_score(rfs, ats, weight=None):
    "log(weighted avg of scores,sqrt(2))"
    if weight is None:
        weight = settings.mention_weight
    avg = (1-weight)*rfs + weight*ats
    if avg<1:
        return 0
    return int(2*log(avg,2))


class Scores(defaultdict):
    def __init__(self):
        defaultdict.__init__(self,int)

    def split(self, uid):
        val = self[uid]
        return val/STATE_FIELD, (val/MAX_SCORE)%MAX_SCORE, val%MAX_SCORE

    def increment(self, uid, rfriends, mentions):
        state, old_rfs, old_ats = self.split(uid)
        rfs = min(old_rfs+(rfriends or 0),MAX_SCORE-1)
        ats = min(old_ats+(mentions or 0),MAX_SCORE-1)
        self[uid] = state*STATE_FIELD + rfs*MAX_SCORE + ats

    def set_state(self, uid, state):
        ignore, rfs, ats = self.split(uid)
        self[uid] = state*STATE_FIELD + rfs*MAX_SCORE + ats

    def dump(self, path):
        with open(path,'w') as f:
            for u,s in self.iteritems():
                print >>f,"%d\t%d"%(u,s)

    def read(self, path):
        with open(path,'r') as f:
            for line in f:
                k,v = [int(s) for s in line.strip().split('\t')]
                self[k]=v

    def count_lookups(self):
        return sum(1 for v in self.itervalues() if v/STATE_FIELD)

