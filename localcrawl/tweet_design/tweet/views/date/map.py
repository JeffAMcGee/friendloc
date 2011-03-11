def map(doc):
    if doc.get('ca'):
        yield doc['ca'][0:3], None
