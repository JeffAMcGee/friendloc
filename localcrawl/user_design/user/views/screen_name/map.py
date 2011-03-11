def map(doc):
  if doc.get('sn'):
    yield doc['sn'], None
