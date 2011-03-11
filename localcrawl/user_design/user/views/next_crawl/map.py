def map(doc):
    if 'ncd' in doc:
        yield doc['ncd'], None
