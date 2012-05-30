import contextlib
import fnmatch
import glob
import inspect
import itertools
import os.path
from collections import defaultdict

import msgpack


def func(gives_keys=False, all_items=False):
    def wrapper(f):
        f.gives_keys = gives_keys
        f.all_items = all_items
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
    """
    everything you need to know about a job when you are not running it
    I may merge this into the function objects @func decorates.
    """
    def __init__(self, func, sources=(), saver='simple_save'):
        self.func = func
        self.saver = saver
        self.sources = sources

    def output_files(self, env):
        if self.split_data:
            return env.glob(self.name+'.*')
        return [self.name,]


class Executor(object):
    def run(self, job, input_paths):
        """
        run the job. save the results. block until it is done.
        """
        raise NotImplementedError


class Storage(object):

    def save(self, name, items):
        "store items in the file 'name'. items is an iterator."
        raise NotImplementedError

    def load(self, name):
        "return an iterator over the items stored in the file 'name'"
        raise NotImplementedError

    def bulk_saver(self, name):
        """
        context manager that returns a BulkSaver object, which can save to
        multiple open files at the same time. The object will have this method:
            def add(self, name, item)
        When that method is called, it will append item to the file with the
        name 'name'.
        """
        raise NotImplementedError

    def glob(self, pattern):
        "return a list of the files that match a shell-style pattern"
        raise NotImplementedError

    def input_paths(self, source_jobs):
        inputs = [job.output_files(self) for job in source_jobs]
        if not all(inputs):
            raise ValueError("missing dependencies for job")
        return itertools.product(*inputs)

    # simple_save, list_reduce, and split?


class SingleThreadExecutor(Executor):
    "Execute in a single process"
    def _runnable_func(self, job):
        if inspect.ismethod(job.func):
            cls = job.func.im_class
            obj = cls(self, self)
            # magic: bind the unbound method
            return job.func.__get__(obj,cls)
        else:
            return job.func

    def _run_single(self, func, in_paths):
        iters = [ self.load(path) for path in in_paths ]
        if func.all_items or not iters:
            return func(*iters)
        else:
            item_iters = itertools.izip_longest(*iters)
            calls = (func(*args) for args in item_iters)
            return chain_truthy(calls)

    def run(self, job, input_paths):
        func = self._runnable_func(job)

        results = {}
        for source_set in input_paths:
            # multiprocess here!
            suffix = ''.join(_chunck(sp) or '' for sp in source_set)
            out_path = job.name+suffix
            assert out_path not in results
            results[out_path] = self._run_single( func, source_set )

        if job.saver:
            getattr(self,job.saver)(job.name,results)

    def simple_save(self, name, results):
        for name, items in results.iteritems():
            self.save(name,items)

    def list_reduce(self, name, results):
        buckets = defaultdict(list)
        # For now, just merge all the results together
        for path,d in results.iteritems():
            for k,v in d:
                buckets[k].append(v)
        # FIXME: What is the best way to pick the name of the merged
        # results?
        self.save(name,buckets.iteritems())

    def split(self, name, results):
        with self.bulk_saver(name) as saver:
            for key,value in results[name]:
                saver.add(_path(name,key),value)


class DictStorage(Storage):
    "Store data in a dict in this class for testing and debugging."
    THE_FS = {}

    def save(self, name, items):
        self.THE_FS[name] = list(items)

    def load(self, name):
        return self.THE_FS[name]

    class BulkSaver(object):
        def __init__(self):
            self.data = defaultdict(list)

        def add(self, name, item):
            self.data[name].append(item)

    @contextlib.contextmanager
    def bulk_saver(self, name):
        bs = DictStorage.BulkSaver()
        yield bs
        for key,items in bs.data.iteritems():
            self.THE_FS[key] = items

    def glob(self, pattern):
        return fnmatch.filter(self.THE_FS,pattern)


class FileStorage(Storage):
    "Store data in a directory"
    def __init__(self, path):
        # path should be an absolute path to a directory to store files
        self.path = path

    def _open(self, name, *args):
        return open(os.path.join(self.path,name), *args)

    def save(self, name, items):
        with self._open(name,'w') as f:
            packer = msgpack.Packer()
            for item in items:
                f.write(packer.pack(item))

    def load(self, name):
        # Can we just return the iterator or is close a problem?
        with self._open(name) as f:
            for item in msgpack.Unpacker(f):
                yield item

    class BulkSaver(object):
        def __init__(self,storage):
            self.files = {}
            self.packer = msgpack.Packer()
            self.storage = storage

        def add(self, name, item):
            if name not in self.files:
                self.files[name] = self.storage._open(name,'w')
            self.files[name].write(self.packer.pack(item))

        def close(self):
            for file in self.files.itervalues():
                file.close()

    @contextlib.contextmanager
    def bulk_saver(self, name):
        bs = self.BulkSaver(self)
        yield bs
        bs.close()

    def glob(self, pattern):
        return glob.glob(os.path.join(self.path,pattern))


class SimpleEnv(DictStorage,SingleThreadExecutor):
    "Run in a single thread, store results in ram"


class SimpleFileEnv(FileStorage,SingleThreadExecutor):
    "Run in a single thread, store results to disk"


class Gob(object):
    def __init__(self, env):
        # Do we want this to be some kind of singleton?
        self.jobs = {}
        self.env = env

    def run_job(self,name):
        job = self.jobs[name]
        source_jobs = [self.jobs[s] for s in job.sources]
        input_paths = self.env.input_paths(source_jobs)
        return self.env.run(job,input_paths=input_paths)

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
        if job.saver == 'split':
            job.split_data = True
        elif job.saver == 'list_reduce':
            job.split_data = False
        elif sources:
            job.split_data = any(self.jobs[s].split_data for s in sources)
        else:
            job.split_data = False

        self.jobs[name] = job
