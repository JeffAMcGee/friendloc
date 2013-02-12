
settings = dict(
    # FIXME: rename region to mongo_db; it is just the database name now
    region = "friendloc",
    db_host = "localhost",
    gisgraphy_url = "http://services.gisgraphy.com",
    mongo_host = 'localhost',
    log_dir = 'logs',
    log_crashes = True,
)

try:
    from settings_local import settings as s
    settings.update(s)
except ImportError:
    pass
