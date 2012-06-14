import contextlib
import fnmatch
import glob
import inspect
import itertools
import os.path
from collections import defaultdict
from multiprocessing import Pool

import msgpack


def mapper(all_items=False):
    def wrapper(f):
        f.all_items = all_items
        return f
    return wrapper


def reducer():
    # purely decorative decorator
    def wrapper(f):
        return f
    return wrapper


@reducer()
def join_reduce(key, items, rereduce):
    if rereduce:
        return tuple(itertools.chain.from_iterable(items))
    else:
        return tuple(items)


@reducer()
def sum_reduce(key, items):
    return sum(items)


def _path(name,key):
    return '.'.join((name,str(key)))


def _call_opt_kwargs(func,*args,**kwargs):
    "Call func with args and kwargs. Omit kwargs func does not understand."
    spec = inspect.getargspec(func)
    if spec.keywords:
        return func(*args,**kwargs)
    safe_kws = {k:v for k,v in kwargs.iteritems() if k in spec.args}
    return func(*args,**safe_kws)


def chain_truthy(iters):
    "takes an iterable of iterators or None values and chains them together"
    non_empty = itertools.ifilter(None, iters)
    return itertools.chain.from_iterable(non_empty)


class Job(object):
    """
    everything you need to know about a job when you are not running it
    """
    def __init__(self, mapper, sources=(), saver='save', reducer=None):
        self.mapper = mapper
        self.saver = saver
        self.sources = sources
        self.reducer = reducer

    def output_files(self, env):
        if self.split_data:
            return env.glob(self.name+'.*')
        return [self.name,]

    def runnable_funcs(self, env):
        """
        if the jobs mapper and reducer are unbound methods, make an object for
        them to bind to, and bind them. If the mapper and reducer are in the
        same class, only make one instance and have them share it.
        """
        res = {}
        if inspect.ismethod(self.mapper):
            cls = self.mapper.im_class
            obj = cls(env)
            # magic: bind the unbound method job.mapper to obj
            res['map'] = self.mapper.__get__(obj,cls)
        else:
            res['map'] = self.mapper

        if inspect.ismethod(self.reducer):
            cls = self.reducer.im_class
            if self.reducer.im_class != self.mapper.im_class:
                # if the classes are the same, we want to re-use the same
                # object, otherwise we make a new one:
                obj = cls(env)
            # magic: bind the unbound method self.mapper to obj
            res['reduce'] = self.reducer.__get__(obj,cls)
        else:
            res['reduce'] = self.reducer
        return res


class Executor(object):
    def run(self, job, input_paths):
        """
        run the job. save the results. block until it is done.

        Subclasses of this should behave like a drop-in replacement for
        SingleThreadExecutor.run(...) .
        """
        raise NotImplementedError

    def map_reduce_save(self, job, in_paths, funcs):
        "completely process one file -> map, reduce, and save"
        results = self.map_single(funcs['map'],in_paths)
        if funcs['reduce']:
            results = self.reduce_single(funcs['reduce'],results)

        self.save_single(job, in_paths, results)
        return results

    def map_single(self, mapper, in_paths):
        "run mapper for one set of inputs and return an iterator of results"
        iters = [ self.load(path) for path in in_paths ]
        if mapper.all_items or not iters:
            return mapper(*iters)
        else:
            item_iters = itertools.izip_longest(*iters)
            calls = (mapper(*args) for args in item_iters)
            return chain_truthy(calls)

    def reduce_single(self, reducer, map_output):
        grouped = defaultdict(list)
        for k,v in map_output:
            grouped[k].append(v)
        return [
            ( key, _call_opt_kwargs(reducer,key,items,rereduce=False) )
            for key,items in grouped.iteritems()
            ]

    def _suffix(self, in_paths):
        def suffix(path):
            name = os.path.basename(path)
            pos = name.find('.')
            return name[pos:] if pos!=-1 else ''

        return ''.join(suffix(sp) for sp in in_paths)

    def save_single(self, job, in_paths, results):
        if job.saver:
            saver = getattr(self,job.saver)
            saver(job.name+self._suffix(in_paths),results)

    def split_save(self, name, key_item_pairs):
        with self.bulk_saver(name) as saver:
            for key,value in key_item_pairs:
                saver.add(_path(name,key),value)

    def reduce_all(self, job, reducer):
        grouped = defaultdict(list)
        for path in self.glob(job.name+'.*'):
            for k,v in self.load(path):
                grouped[k].append(v)
        results = [
            ( key, _call_opt_kwargs(reducer,key,items,rereduce=True) )
            for key,items in grouped.iteritems()
            ]
        self.save(job.name,results)


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


class SingleThreadExecutor(Executor):
    "Execute in a single process"
    def run(self, job, input_paths):
        """
        This is the cannonical implementation of run for an Executor.
        Subclasses of executor should be roughly equivalent to this method, but
        they can run map_reduce_save in different processes or on different
        machines.

        runnable_funcs should be called once per proccess so that the instances
        of the method that contains the mapper and reducer can be used to cache
        data.
        """
        funcs = job.runnable_funcs(self)

        for source_set in input_paths:
            self.map_reduce_save(job, source_set, funcs)

        if funcs['reduce']:
            self.reduce_all(job, funcs['reduce'])


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
        self.path = os.path.abspath(path)

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


def _mp_worker_init(env, job):
    MultiProcEnv._worker_data['env'] = env
    MultiProcEnv._worker_data['job'] = job


def _mp_worker_run(source_set):
    # The map method must be a top-level method, not a class method because
    # pickle is broken.
    env = MultiProcEnv._worker_data['env']
    job = MultiProcEnv._worker_data['job']
    funcs = MultiProcEnv._worker_data['funcs']
    env.map_reduce_save(job, source_set, funcs)


class MultiProcEnv(FileStorage, Executor):
    "Use python's multiprocessing to split up a job"

    _worker_data = {}

    def run(self, job, input_paths):
        MultiProcEnv._worker_data['funcs'] = job.runnable_funcs(self)

        pool = Pool(4,_mp_worker_init,[self, job])
        pool.map(_mp_worker_run, input_paths)
        pool.close()
        pool.join()

        reducer = MultiProcEnv._worker_data['funcs']['reduce']
        if reducer:
            self.reduce_all(job, reducer)

        MultiProcEnv._worker_data = {}


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

    def add_job(self, mapper, sources=(), *args, **kwargs):
        if isinstance(sources,basestring):
            sources = (sources,)

        name = kwargs.get('name',mapper.__name__.lower())
        if name in self.jobs:
            raise ValueError('attempt to insert second job with same name')

        for source in sources:
            if source not in self.jobs:
                raise LookupError('sources must be defined first')

        job = Job(mapper,sources,*args,**kwargs)
        # XXX: I don't like mucking with job here
        job.name = name
        if job.saver == 'split_save':
            job.split_data = True
        elif job.reducer:
            job.split_data = False
        elif sources:
            job.split_data = any(self.jobs[s].split_data for s in sources)
        else:
            job.split_data = False

        self.jobs[name] = job
