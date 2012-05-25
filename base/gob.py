import inspect
import itertools
import functools
from collections import defaultdict


def func(gives_keys=False,all_items=False):
    def wrapper(f):
        f.gives_keys = gives_keys
        f.all_items = all_items
        return f
    return wrapper

#XXX
THE_FS = {}

class Job(object):
    """
    Abstract base class representing a job to run.
    """
    def __init__(self, func, sources, saver=None):
        self.func = func
        self.saver = saver or Job.simple_save
        self.sources = sources

    def run(self):
        if inspect.ismethod(self.func):
            cls = self.func.im_class
            obj = cls(self)
            # magic: bind the unbound method
            func = self.func.__get__(obj,cls)
        else:
            func = self.func

        if self.sources:
            items = THE_FS[self.sources[0]]
            if func.all_items:
                results = func(items)
            else:
                calls = (func(item) for item in items)
                non_empty = itertools.ifilter(None, calls)
                results = itertools.chain.from_iterable(non_empty)
        else:
            results = func()

        # FIXME: smarter way to handle no output?
        if results:
            self.saver(self,'nothing',self.name,results)

    def simple_load(self, dir, name):
        return THE_FS[name]

    def simple_save(self, dir, name, items):
        THE_FS[name] = list(items)

    def list_reduce(self, dir, name, items):
        results = defaultdict(list)
        for k,v in items:
            results[k].append(v)
        THE_FS[name] = results

    def split(self, dir, name, items):
        THE_FS[name] = [item[1] for item in items]


class Gob(object):
    def __init__(self):
        # Do we want this to be some kind of singleton?
        self.jobs = {}
        self.directory = 'data'

    def run_job(self,name):
        return self.jobs[name].run()

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
