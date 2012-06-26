import json
import re

from restkit import Resource

from settings import settings
from models import GeonamesPlace
from utils import in_local_box

class GisgraphyResource(Resource):
    COORD_RE = re.compile('(-?\d+\.\d+), *(-?\d+\.\d+)')

    def __init__(self):
        Resource.__init__(self,
                settings.gisgraphy_url,
                client_opts={'timeout':30},
        )
        self._mdist = {}

    def set_mdists(self,mdists):
        self._mdist = mdists

    def mdist(self,gnp):
        id = str(gnp.feature_id)
        if id in self._mdist:
            return self._mdist[id]
        if gnp.feature_code in self._mdist:
            return self._mdist[gnp.feature_code]
        return self._mdist.get('other',None)

    def fulltextsearch(self, q, headers=None, **kwargs):
        #we make the query lower case as workaround for "Portland, OR"
        r = self.get('fulltext/fulltextsearch',
            headers,
            q=q,
            format="json",
            spellchecking=False,
            **kwargs)
        return json.loads(r.body_string())["response"]["docs"]

    def twitter_loc(self, q):
        if not q: return None
        # check for "30.639, -96.347" style coordinates
        match = self.COORD_RE.search(q)
        if match:
            return GeonamesPlace(
                lat=float(match.group(1)),
                lng=float(match.group(2)),
                feature_code='COORD',
                mdist=self._mdist.get('COORD',None),
            )
        #try gisgraphy
        q = q.lower().strip().replace('-','/').replace(',',', ')
        q = ''.join(re.split('[|&!+]',q))
        if not q: return None
        results = self.fulltextsearch(q)
        # otherwise, return the best result
        if results:
            res = GeonamesPlace(results[0])
            res.mdist = self.mdist(res)
            return res
        # try splitting q in half
        found = None
        for splitter in ('and','or','/'):
            parts = q.split(splitter)
            if len(parts)==2:
                for part in parts:
                    res = self.twitter_loc(part)
                    if res and in_local_box(res.to_d()):
                        return res
                    if res:
                        found = res
        return found

if __name__ == '__main__':
    res = GisgraphyResource()
    f = res.fulltextsearch('Austin TX')
