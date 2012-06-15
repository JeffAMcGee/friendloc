import math
from itertools import chain

from sklearn import preprocessing, tree, cross_validation
import numpy as np

from base import gob
from base.utils import coord_in_miles

def logify(x):
    return int(math.ceil(math.log(x+1,2)))

@gob.mapper()
def edge_vect(user):

    dists = [ logify(coord_in_miles(user['mloc'],rel)) for rel in user['rels'] ]
    # including the dist in the median is broken.
    fol_dist = int(np.median(dists))

    for dist,rel in zip(dists,user['rels']):
        flags = [rel['kind'] >>i & 1 for i in range(4)]
        vals = [logify(rel[k]) for k in ('mdist','folc')]
        yield flags+vals+[fol_dist, dist]


def _transformed(vects):
    # convert vects to a scaled numpy array
    vects_ = np.fromiter( chain.from_iterable(vects), int )
    vects_.shape = (len(vects_)//8),8
    X = np.array(vects_[:,:-1],float)
    y = np.array(vects_[:,-1],int)
    scaler = preprocessing.Scaler().fit(X)

    # In final system, you will want to only fit the training data.
    return scaler.transform(X), y


@gob.mapper(all_items=True)
def fl_learn(vects):
    #./gb.py -i edge_vect.22 fl_learn
    X, y = _transformed(vects)
    clf = tree.DecisionTreeClassifier()
    scores = cross_validation.cross_val_score(clf,X,y,cv=5)
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
    return []

