def map(doc):
    if 'rfs' in doc and 'ats' in doc:
        from math import log
        avg = (doc['rfs'] + doc['ats']) /2.0
	yield int(log(max(avg,1),2)),None
