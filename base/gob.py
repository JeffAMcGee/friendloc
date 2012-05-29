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


def _chunck(path):
    pos = path.find('.')
    return path[pos:] if pos!=-1 else None


def chain_truthy(iters):
    non_empty = itertools.ifilter(None, iters)
    return itertools.chain.from_iterable(non_empty)


class Job(object):
    def __init__(self, func, sources=(), saver=None):
        self.func = func
        self.saver = saver or Job.simple_save
        self.sources = sources

    def _runnable_func(self, storage):
        if inspect.ismethod(self.func):
            cls = self.func.im_class
            obj = cls(self, storage)
            # magic: bind the unbound method
            return self.func.__get__(obj,cls)
        else:
            return self.func

    def _run_single(self, storage, func, in_paths):
        """

        If the func has all_items set to false, iters should contain things you
        can re-iterate over like tuples.
        """
        iters = [ storage.load(path) for path in in_paths ]
        if func.all_items or not iters:
            return func(*iters)
        else:
            item_iters = itertools.izip_longest(*iters)
            calls = (func(*args) for args in item_iters)
            return chain_truthy(calls)

    def output_files(self, storage):
        if self.split_data:
            return storage.glob(self.name+'.*')
        return [self.name,]

    def run(self, storage, source_jobs):
        func = self._runnable_func(storage)

        inputs = [job.output_files(storage) for job in source_jobs]
        if not all(inputs):
            raise ValueError("missing dependencies for job")

        source_sets = itertools.product(*inputs)
        results = {}

        for source_set in source_sets:
            # multiprocess here!
            suffix = ''.join(_chunck(sp) or '' for sp in source_set)
            out_path = self.name+suffix
            assert out_path not in results
            results[out_path] = self._run_single( storage, func, source_set )

        if func.must_output:
            assert any(results.itervalues())
        self.saver(self,storage,results)

    def simple_save(self, storage, results):
        for name, items in results.iteritems():
            storage.save(name,items)

    def list_reduce(self, storage, results):
        results = defaultdict(list)
        for k,v in items:
            results[k].append(v)
        storage.save(name,results.iteritems())

    def split(self, storage, results):
        with storage.bulk_saver(self.name) as saver:
            for key,value in results[self.name]:
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
        cleaned = pattern.replace('.','\\.').replace('?','.').replace('*','.*')
        regex = re.compile(cleaned)
        return [k for k in self.THE_FS if regex.match(k)]


class Gob(object):
    def __init__(self):
        # Do we want this to be some kind of singleton?
        self.jobs = {}
        self.storage = DictStorage()

    def run_job(self,name):
        job = self.jobs[name]
        source_jobs = [self.jobs[s] for s in job.sources]
        return job.run(self.storage, source_jobs=source_jobs)

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
            job.split_data = any(self.jobs[s].split_data for s in sources)
        else:
            job.split_data = False

        self.jobs[name] = job
