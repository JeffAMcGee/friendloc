def map(doc):
    if doc.get('geo'):
        yield None, None
