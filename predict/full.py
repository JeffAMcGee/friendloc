
from base.models import User
from base import gob, twitter, gisgraphy
from explore import sprawl
from predict import fl


def _crawl_pred_one(user,twit,gis,pred):
    # FIXME: check if this user's location is already in the db?
    if user.location and not user.geonames_place:
        user.geonames_place = gis.twitter_loc(user.location)

    gnp = user.geonames_place
    if gnp and gnp.mdist<25:
        user.pred_loc = gnp.to_tup()
        return

    nebrs, ats, ated = sprawl.crawl_single(user,twit,gis)
    user.pred_loc = pred.predict( user, nebrs, ats, ated )


@gob.mapper(all_items=True,slurp={'mdists':next})
def crawl_predict(user_ds, env, mdists):
    """
    takes a user dictionary, runs the crawler and predictor
    """
    twit = twitter.TwitterResource()
    gis = gisgraphy.GisgraphyResource()
    gis.set_mdists(mdists)

    pred = fl.Predictors(env)
    pred.load_env(env,'0')
    for user_d in user_ds:
        user = User(user_d)
        _crawl_pred_one(user,twit,gis,pred)
        yield user.to_d()

