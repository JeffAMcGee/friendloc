
from friendloc.base.models import User
from friendloc.base import gob, twitter, gisgraphy
from friendloc.explore import sprawl
from friendloc.predict import fl

# FIXME: this should be a setting
# this is the worst PLE that we are willing to accept
MAX_GNP_MDIST = 25


class FriendlyLocator(object):
    def __init__(self, env=None, path=None, settings=None):
        """
        """
        if not env or not path:
            raise ValueError('FriendlyLocator needs a path or an env')

        if env:
            self.env = env
        else:
            self.env = gob.SimpleFileEnv(path,log_crashes=True)

        mdists = next(self.env.load('mdists'))
        self.twit = twitter.TwitterResource(settings)
        self.gis = gisgraphy.GisgraphyResource()
        self.gis.set_mdists(mdists)
        self.pred = fl.Predictors(env)
        self.pred.load_env(env,'0')

    def predict(self, user_d, steps=0):
        """
        Attept to locate a Twitter user.
            user_d should be a Twitter-style user dictionary
            steps is the number of steps on the social graph to crawl. It should
                be 0, 1, or 2. If 0, predict makes no Twitter API calls, 1 uses
                4 calls, and 2 uses around 80 API calls.
        returns (longitude, latitude) or None if no location can be found
        """
        user = User(user_d)
        if steps==0 and not user.location:
            return None

        gnp = self.gis.twitter_loc(user.location)

        if steps==0:
            return gnp.to_tup() if gnp else None

        if gnp and gnp.mdist<MAX_GNP_MDIST:
            user.geonames_place = gnp
            return gnp.to_tup()

        _crawl_pred_one(user,self.twit,self.gis,self.pred,fast=(steps==1))
        return user.pred_loc


def _crawl_pred_one(user,twit,gis,pred,fast):
    if user.location and not user.geonames_place:
        user.geonames_place = gis.twitter_loc(user.location)

    gnp = user.geonames_place
    if gnp and gnp.mdist<MAX_GNP_MDIST:
        user.pred_loc = gnp.to_tup()
        return

    nebrs, ats, ated = sprawl.crawl_single(user,twit,gis,fast=fast)
    if nebrs:
        clf = 'friendloc_nearcut' if fast else 'friendloc_cut'
        user.pred_loc = pred.predict( user, nebrs, ats, ated, clf)


@gob.mapper(all_items=True)
def cheap_predict(user_ds, env):
    """
    takes a user dictionary, runs the geocoder without crawling, adds the
    location if we can find one
    """
    return crawl_predict(user_ds, env, 0)


@gob.mapper(all_items=True)
def crawl_predict_fast(user_ds, env, mdists):
    """
    takes a user dictionary, runs the crawler and predictor without using
    information from leafs or edges of contacts
    """
    return crawl_predict(user_ds, env, 1)


@gob.mapper(all_items=True)
def crawl_predict(user_ds, env, steps=2):
    """
    takes a user dictionary, runs the crawler and predictor using information
    from leafs
    """
    locator = FriendlyLocator()
    for user_d in user_ds:
        yield locator.predict(user_d,steps=steps)

