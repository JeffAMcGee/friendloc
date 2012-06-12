from collections import defaultdict
from operator import itemgetter

from settings import settings
from base.models import *
from base.utils import *


def _haversine(lng1, lng2, lat1, lat2):
    dlng = lng2 - lng1
    dlat = lat2 - lat1
    a = numpy.sin(dlat/2)**2 + numpy.cos(lat1)*numpy.cos(lat2)*numpy.sin(dlng/2)**2
    c = 2 * numpy.arctan2(numpy.sqrt(a), numpy.sqrt(1-a)) 
    return 3959 * c

def _calc_dists(rels):
    lats = [math.radians(r['lat']) for r in rels]
    lngs = [math.radians(r['lng']) for r in rels]
    lat1,lat2 = numpy.meshgrid(lats,lats)
    lng1,lng2 = numpy.meshgrid(lngs,lngs)
    return _haversine(lng1,lng2,lat1,lat2)


def eval_block(prefix, block):
    predictors = dict(
        diff= [
            ('FriendlyLocation', FriendlyLocation(True,True)),
            ('Mode', Mode()),
        ],
        samp= [
            ('Mode', Mode()),
            ('Median', Median()),
            ('FriendlyLocation (Location Error)', FriendlyLocation(False,True)),
            ('FriendlyLocation (Simple)', FriendlyLocation(False,False)),
        ],
        pick= [
            ('Omniscient *', Omniscient()),
            ('FriendlyLocation (Full) *', FriendlyLocation(True,True)),
            ('FriendlyLocation (Relationship Types) *', FriendlyLocation(True,False)),
        ],
        lul= [
            ('a 0<=lul<=49', FriendlyLocation(True,True,min_lul=0,max_lul=50)),
            ('b 50<=lul<=99', FriendlyLocation(True,True,min_lul=50,max_lul=100)),
            ('c 100<=lul<=199', FriendlyLocation(True,True,min_lul=100,max_lul=200)),
            ('d 200<=lul<=399', FriendlyLocation(True,True,min_lul=200,max_lul=400)),
            ('e 400<=lul', FriendlyLocation(True,True,min_lul=400)),
        ],
        top= [
            ('a top 5', FriendlyLocation(True,True,min_lul=200,top=5)),
            ('b top 10', FriendlyLocation(True,True,min_lul=200,top=10)),
            ('c top 25', FriendlyLocation(True,True,min_lul=200,top=25)),
            ('d top 100', FriendlyLocation(True,True,min_lul=200,top=100)),
            ('e all', FriendlyLocation(True,True,min_lul=200,top=400)),
        ],
    ) 
    users = read_json("data/pick"+block)
    dists = defaultdict(list)
    skipped=0

    for i,user in enumerate(users):
        mloc = user['mloc']
        if not user['rels']:
            skipped+=1
            continue
        user['dists'] = _calc_dists(user['rels'])
        if 'gnp' not in user or user['gnp']['code']=="COORD":
            user['gnp'] = settings.default_loc
        for label, predictor in predictors[prefix]:
            res = predictor.pred(user)
            if res is not None:
                dists[label].append(coord_in_miles(mloc, res))
    write_json([dists],"data/%s_res%s"%(prefix,block))
    print "saved %s"%block


class Mode():
    def pred(self, user):
        counts = defaultdict(int)
        for r in user['rels']:
            counts[(r['lng'],r['lat'])]+=1
        spot, count =  max(counts.iteritems(),key=itemgetter(1))
        return spot

class Median():
    def pred(self, user):
        return median_2d((r['lng'],r['lat']) for r in user['rels'])


class Omniscient():
    def pred(self, user):
        best = min(user['rels'], key=lambda rel: coord_in_miles(user['mloc'],rel))
        return best


class FriendlyLocation():
    def __init__(self, use_params, use_mdist, min_lul=None, max_lul=None, top=2000):
        self.use_mdist = use_mdist
        self.use_params = use_params
        self.min_lul = min_lul
        self.max_lul = max_lul
        self.top = top
        if use_params:
            params = list(read_json("params"))[0]
            for k,v in params.iteritems():
                setattr(self,k,v)

    def pred(self, user):
        if self.max_lul is not None and user['lu_len']>= self.max_lul:
            return None
        if self.min_lul is not None and user['lu_len']< self.min_lul:
            return None

        self.rels = user['rels'][:self.top]
        buckets = settings.fol_count_buckets
        for rel in self.rels:
            if self.use_params:
                kind = rel['kind']
                folc = min(buckets-1, int(math.log(max(rel['folc'],1),4)))
                rel['inner'] = self.inner[kind][folc]
                rel['a'] = self.a[kind][folc]
                rel['e_b'] = math.e**self.b[kind][folc]
                rel['rand'] = self.rand[kind][folc]/2750
            else:
                rel['inner'] = .133
                rel['a'] = -1.423
                rel['e_b'] = math.e**-2.076
                rel['rand'] = .568/2750
            rel['md_fixed'] = max(10,rel['mdist']) if self.use_mdist else 10

        best, unused = max(zip(self.rels,user['dists']), key=self._user_prob)
        return best

    def _user_prob(self, rel_dists):
        rel, dists = rel_dists
        return sum(self._edge_prob(edge,dist) for edge,dist in zip(self.rels,dists))

    def _edge_prob(self, edge, dist):
        mdist = edge['md_fixed']
        local = edge['inner'] if dist<mdist else edge['e_b'] * ((dist/mdist)**edge['a']) 
        res =  math.log(local/mdist+edge['rand'])
        return res
