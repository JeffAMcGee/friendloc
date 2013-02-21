friendloc
=========

a tool for estimating the locations of users on twitter

LICENSE
=======

This falls under the Apache License, Version 2.0
Get a copy at http://www.apache.org/licenses/LICENSE-2.0.html

REQUIREMENTS
------------
* python 2.7
* the python packages listed in reqs.pip
* Gisgraphy (http://www.gisgraphy.com/)
* data from Twitter's streaming API
* a Twitter account and application to access the REST API


GETTING STARTED
---------------


You'll need to create a `settings_dev.py` or `settings_prod.py` in order to run
localcrawl.  The easiest way to do this is to copy settings_dev.py.template to
settings_dev.py.

    cp settings_dev.py.template settings_dev.py

You will also need to go through the three-legged oauth to get the information
that goes in settings_dev.py .  This will help you:
    http://benoitc.github.com/restkit/authentication.html#oauth

All the data you need to do location prediction is here:
http://infolab.tamu.edu/static/users/jeff/friendloc_data.tgz

Starting
----------

To get the crawling process started you will need tweets from Twitter's
streaming API.  I used @bde's https://github.com/bde/TwitterStreamSaver to
collect them.

Once you have the gzipped tweets, you can pick the users to crawl with the
following command:

    gunzip -c ~/may/*/*.gz | ./gb.py -s mloc_users

localcrawl
----------

The lo-calorie twitter neighborhood crawler.

This was started as a seperate project, but it is now contained in the
localcrawl subdirectory. It is probably broken.

Nomenclature
------------
In the process of writing the FriendlyLocation paper, I renamed a few things.
This code still uses the old names.

paper                   | code
++++++++++++++++++++++++++++++++++++++++++++++++
target users            | mloc users
home location           | mloc, median_loc
contact                 | neighbor
median location error   | mdist
pContact                | vect_ratios
