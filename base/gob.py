import inspect
import itertools
from collections import defaultdict


def func(gives_keys=False, all_items=False, must_output=True):
    def wrapper(f):
        f.gives_keys = gives_keys
        f.all_items = all_items
        f.must_output = must_output
        return f
    return wrapper


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
        storage.save(name, [item[1] for item in items])


class Storage():
    #XXX: Storage will become an interface, this implementation is a mock
    THE_FS = {}

    def save(self, name, items):
        self.THE_FS[name] = list(items)

    def load(self, name):
        return self.THE_FS[name]


class Gob(object):
    def __init__(self):
        # Do we want this to be some kind of singleton?
        self.jobs = {}
        self.storage = Storage()

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
        # XXX
        job.name = name

        self.jobs[name] = job
