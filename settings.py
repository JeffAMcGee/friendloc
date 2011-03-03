import pdb
class SettingsBunch(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

settings = SettingsBunch(
    local_box = dict(lat=(29,30.5),lng=(-96,-94.5)),
    region = "houtx",
    gisgraphy_url = "http://services.gisgraphy.com",
    beanstalk_host = 'localhost',
    beanstalk_port = 11300,
    couchdb_root = 'http://localhost:5984/',
    beanstalkd_ttr = 3600,
    pdb = pdb.set_trace,
)

try:
    from settings_prod import settings as s
except:
    from settings_dev import settings as s
settings.update(s)
