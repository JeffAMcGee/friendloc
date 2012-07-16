import math
from itertools import chain

from sklearn import preprocessing, tree, cross_validation
import numpy as np

from base import gob
from base.utils import coord_in_miles

def logify(x,fudge=1):
    return math.log(x+1,2)

def _scaled_local(x):
    return x if x is not None else .3

@gob.mapper()
def nebr_vect(user):
    for nebr in user['nebrs']:
        flags = [nebr['kind'] >>i & 1 for i in range(3)]
        logged = [logify(nebr[k]) for k in ('mdist','folc','frdc')]
        others = [
            _scaled_local(nebr['lofrd']),
            _scaled_local(nebr['lofol']),
            int(bool(nebr['prot'])),
            logify(coord_in_miles(user['mloc'],nebr),fudge=.01),
        ]
        yield flags + logged + others


def _transformed(vects):
    # convert vects to a scaled numpy array
    vects_ = np.fromiter( chain.from_iterable(vects), np.float32 )
    vects_.shape = (len(vects_)//10),10
    X = np.array(vects_[:,:-1])
    y = np.array(vects_[:,-1])
    #scaler = preprocessing.Scaler().fit(X)

    # In final system, you will want to only fit the training data.
    #return scaler.transform(X), y
    return X,y


@gob.mapper(all_items=True)
def fl_learn(vects):
    #./gb.py -i edge_vect.22 fl_learn
    X, y = _transformed(vects)
    clf = tree.DecisionTreeRegressor(max_depth=8)
    clf.fit(X,y)
    import ipdb; ipdb.set_trace()
    scores = cross_validation.cross_val_score(clf,X,y,cv=5)
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
    return []

