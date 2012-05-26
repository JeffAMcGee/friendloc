import contextlib
import inspect
import itertools
import re
from collections import defaultdict


def func(gives_keys=False, all_items=False, must_output=True):
    def wrapper(f):
        f.gives_keys = gives_keys
        f.all_items = all_items
        f.must_output = must_output
        return f
    return wrapper


def _path(name,key):
    return '.'.join((name,str(key)))


class Job(object):
    def __init__(self, func, sources, saver=None):
        self.func = func
        self.saver = saver or Job.simple_save
        self.sources = sources

    def run(self, storage):
        if inspect.ismethod(self.func):
            cls = self.func.im_class
            obj = cls(self, storage)
            # magic: bind the unbound method
            func = self.func.__get__(obj,cls)
        else:
            func = self.func

        if self.sources:
            items = storage.load(self.sources[0])
            if func.all_items:
                results = func(items)
            else:
                calls = (func(item) for item in items)
                non_empty = itertools.ifilter(None, calls)
                results = itertools.chain.from_iterable(non_empty)
        else:
            results = func()

        if func.must_output or results:
            self.saver(self,storage,self.name,results)

    def simple_save(self, storage, name, items):
        storage.save(name,items)

    def list_reduce(self, storage, name, items):
        results = defaultdict(list)
        for k,v in items:
            results[k].append(v)
        storage.save(name,results.iteritems())

    def split(self, storage, name, items):
        with storage.bulk_saver(name) as saver:
            for key,value in items:
                saver.add(key,value)


class DictStorage(object):
    THE_FS = {}

    def save(self, name, items):
        self.THE_FS[name] = list(items)

    def load(self, name):
        return self.THE_FS[name]

    class BulkSaver(object):
        def __init__(self):
            self.data = defaultdict(list)

        def add(self, key, item):
            self.data[key].append(item)

    @contextlib.contextmanager
    def bulk_saver(self, name):
        bs = DictStorage.BulkSaver()
        yield bs
        for key,items in bs.data.iteritems():
            DictStorage.THE_FS[_path(name,key)] = items

    def bulk_loader(self, name):
        """ concatenate several files together """
        # do we even need this?
        for path in self.glob(name):
            for item in self.load(path):
                yield item

    def glob(self, pattern):
        # FIXME: this isn't perfectly compatible with shell globs...
        if pattern.endswith('*'):
            pattern = pattern.rstrip('*')
        else:
            pattern = pattern + '$'
        regex = re.compile(pattern.replace('?','.').replace('*','.*'))
        return [k for k in self.THE_FS if regex.match(k)]


class Gob(object):
    def __init__(self):
        # Do we want this to be some kind of singleton?
        self.jobs = {}
        self.storage = DictStorage()

    def run_job(self,name):
        return self.jobs[name].run(self.storage)

    def add_job(self, func, sources=(), *args, **kwargs):
        if isinstance(sources,basestring):
            sources = (sources,)

        name = kwargs.get('name',func.__name__.lower())
        if name in self.jobs:
            raise ValueError('attempt to insert second job with same name')

        for source in sources:
            if source not in self.jobs:
                raise LookupError('sources must be defined first')

        job = Job(func,sources,*args,**kwargs)
        # XXX: I don't like mucking with job here
        job.name = name
        if job.saver == Job.split:
            job.split_data = True
        elif job.saver == Job.list_reduce:
            job.split_data = False
        elif sources:
            job.split_data = self.jobs[sources[0]].split_data
        else:
            job.split_data = False

        self.jobs[name] = job
