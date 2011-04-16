#!/usr/bin/env python

if __name__ != '__main__':
    print """
This is a tool for testing and administrative tasks.  It is designed to
be %run in ipython or from the command line.  If you import it from another
module, you're doing something wrong.  
"""

import logging
import sys
import getopt
import pdb

try:
    import beanstalkc
except:
    beanstalkc = None
import maroon

from settings import settings
import base.twitter as twitter
from base.gisgraphy import GisgraphyResource
from maroon import *
from base.models import *
from localcrawl.peek import *
from localcrawl.admin import *
from explore.graph import *
from explore.peek import *
from base.utils import *

logging.basicConfig(level=logging.INFO)

if len(sys.argv)>1:
    try:
        opts, args = getopt.getopt(sys.argv[2:], "c:m:s:e:h:")
    except getopt.GetoptError, err:
        print str(err)
        print "usage: ./admin.py function_name [-c couchdb] [-m mongodb] [-s startkey] [-e endkey] [arguments]"
        sys.exit(2)
    kwargs={}
    for o, a in opts:
        if o == "-h":
            if getattr(Model,'database',None):
                raise Exception("-h must come before -c or -m")
            settings.db_host=a
        elif o == "-c":
            Model.database = couch(a)
        elif o == "-m":
            Model.database = mongo(a)
        elif o == "-s":
            kwargs['start']=a
        elif o == "-e":
            kwargs['end']=a
        elif o !='-h':
            raise Exception("unhandled option")
    try:
        locals()[sys.argv[1]](*args,**kwargs)
    except:
        logging.exception("command failed")
        pdb.post_mortem()
else:
    gisgraphy = GisgraphyResource()
    twitter = twitter.TwitterResource()
    Model.database = mongo(settings.region)
    try:
        stalk = beanstalkc.Connection()
    except:
        pass
