import contextlib
import inspect
import itertools
import logging
import sys
import os.path
import traceback
from collections import defaultdict
from multiprocessing import Pool

import msgpack


class StatefulIter(object):
    "wrap an iterator and keep the most recent value in self.current"
    NOT_STARTED = object()
    FINISHED = object()

    def __init__(self,it):
        self.it = iter(it)
        self.current = self.NOT_STARTED

    def __iter__(self):
        return self

    def next(self):
        try:
            self.current = next(self.it)
        except StopIteration:
            self.current = self.FINISHED
            raise
        return self.current


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
def set_reduce(key, items, rereduce):
    if rereduce:
        return tuple(set(itertools.chain.from_iterable(items)))
    else:
        return tuple(set(items))


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


class Job(object):
    """
    everything you need to know about a job when you are not running it
    """
    def __init__(self, mapper=None, sources=(), saver='save', reducer=None,
                 name=None, split_data=False, procs=6):
        self.mapper = mapper
        self.saver = saver
        self.sources = sources
        self.reducer = reducer
        self.name = name
        self.procs = procs
        self.split_data = split_data

    def output_files(self, env):
        if self.split_data:
            return env.split_files(self.name)
        return [self.name,]

    def load_output(self, path, env):
        "load the results of a previous run of this job"
        return env.load(path)

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


class Cat(Job):
    def load_output(self, path, env):
        job = self.sources[0]
        loads = (job.load_output(p,env) for p in job.output_files(env))
        return itertools.chain.from_iterable(loads)


class Source(Job):
    def __init__(self, source_func=None, **kwargs):
        super(Source,self).__init__(**kwargs)
        self.source_func = source_func

    def load_output(self,path,env):
        return self.source_func()


class Executor(object):
    def __init__(self, log_crashes=False, log_path="logs", log_level=None,
                 **kwargs):
        super(Executor,self).__init__(**kwargs)
        self.log_crashes = log_crashes

        self._file_hdlr = None
        self._log_path = log_path
        self._log_level = log_level

    def setup_logging(self, label):
        if self._log_level is None:
            return
        root = logging.getLogger()
        if len(root.handlers)==0:
            # send logs to stdout if they aren't already headed there
            hdlr = logging.StreamHandler()
            root.addHandler(hdlr)
            hdlr.setLevel(self._log_level)
            root.setLevel(self._log_level)
        if self._log_path is not None:
            filepath = os.path.join(self._log_path, label)
            file_hdlr = logging.FileHandler(filepath, 'a')
            fmt = logging.Formatter(
                "%(levelname)s:%(module)s:%(asctime)s:%(message)s",
                "%H:%M:%S")
            file_hdlr.setFormatter(fmt)
            root.addHandler(file_hdlr)
            file_hdlr.setLevel(self._log_level)
            self._file_hdlr = file_hdlr

    def stop_logging(self):
        if self._file_hdlr:
            logging.getLogger().removeHandler(self._file_hdlr)


    def run(self, job, input_paths):
        """
        run the job. save the results. block until it is done.

        Subclasses of this should behave like a drop-in replacement for
        SingleThreadExecutor.run(...) .
        """
        raise NotImplementedError

    def map_reduce_save(self, job, in_paths, funcs):
        "completely process one file -> map, reduce, and save"
        assert len(job.sources) == len(in_paths)
        inputs = [
                src.load_output(path, self)
                for src, path in zip(job.sources,in_paths)
                ]
        results = self.map_single(funcs['map'],inputs)
        if funcs['reduce']:
            results = self.reduce_single(funcs['reduce'],results)

        self.save_single(job, in_paths, results)
        return results

    def map_single(self, mapper, inputs):
        "run mapper for one set of inputs and return an iterator of results"
        all_items = mapper.all_items or not inputs
        if all_items:
            sfi_inputs = [StatefulIter(it) for it in inputs]
            item_iters = [ sfi_inputs ]
        else:
            item_iters = itertools.izip_longest(*inputs)

        for args in item_iters:
            try:
                results = mapper(*args)
                if results:
                    for res in results:
                        yield res
            except:
                if all_items:
                    current_args = [it.current for it in sfi_inputs]
                else:
                    current_args = args
                self._handle_err(mapper,current_args)
                raise

    def _handle_err(self, mapper, args):
        msg = "map %r failed for %r"%(mapper,args)
        logging.exception(msg)
        if self.log_crashes:
            with open('gobstop.%s.%d'%(mapper.__name__,os.getpid()),'w') as gs:
                print >>gs,msg
                traceback.print_exc(file=gs)


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
        else:
            # force the generator to generate
            for res in results:
                pass

    def split_save(self, name, key_item_pairs):
        with self.bulk_saver(name) as saver:
            for key,value in key_item_pairs:
                saver.add(_path(name,key),value)

    def reduce_all(self, job, reducer):
        grouped = defaultdict(list)
        for path in self.split_files(job.name):
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

    def split_files(self, name):
        "return an iterator of the names that name was split into"
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

    def _handle_err(self, mapper, args):
        msg = "map %r failed for %r"%(mapper,args)
        logging.exception(msg)
        print msg
        traceback.print_exc()
        import ipdb
        ipdb.post_mortem(sys.exc_info()[2])


class DictStorage(Storage):
    "Store data in a dict in this class for testing and debugging."
    THE_FS = {}

    def save(self, name, items):
        self.THE_FS[name] = list(items)

    def load(self, name):
        return iter(self.THE_FS[name])

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

    def split_files(self, name):
        return (n for n in self.THE_FS if n.startswith(name+'.'))


class FileStorage(Storage):
    "Store data in a directory"
    def __init__(self, path, **kwargs):
        super(FileStorage,self).__init__(**kwargs)
        # path should be an absolute path to a directory to store files
        self.path = os.path.abspath(path)

    def _open(self, name, mode='r'):
        parts = name.split('.')
        path = os.path.join(self.path,*parts)+'.mp'
        dir = os.path.dirname(path)
        if 'w' in mode:
            try:
                os.makedirs(dir)
            except OSError:
                # attempt to make it and then check to avoid race condition
                if not os.path.exists(dir):
                    raise

        return open(path,mode)

    def save(self, name, items):
        with self._open(name,'w') as f:
            packer = msgpack.Packer()
            for item in items:
                f.write(packer.pack(item))

    def load(self, name):
        # Can we just return the iterator or is close a problem?
        with self._open(name) as f:
            for item in msgpack.Unpacker(f,encoding='utf-8'):
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

    def split_files(self, name):
        for dirpath, dirs, files in os.walk(os.path.join(self.path,name)):
            _dir = dirpath.replace(self.path,'').lstrip('/').replace('/','.')
            for file in files:
                if file.endswith(".mp"):
                    yield "%s.%s"%(_dir,file[:-3])


def _mp_worker_init(env, job):
    MultiProcEnv._worker_data['env'] = env
    MultiProcEnv._worker_data['job'] = job
    env.setup_logging('%s.%d'%(job.name,os.getpid()))


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

        pool = Pool(job.procs,_mp_worker_init,[self, job])
        pool.map(_mp_worker_run, input_paths, chunksize=1)
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
        input_paths = self.env.input_paths(job.sources)
        return self.env.run(job,input_paths=input_paths)

    def add_cat(self, name, source):
        """ concatenate a split source into one source """
        return self.add_job(sources=(source,), name=name, job_cls=Cat)

    def add_source(self, source, name=None):
        """
        Adds a source that the you, the library user, create. This can be used
        to provide starting data.
        """
        if not name:
            name = source.__name__.lower()
        return self.add_job(source_func=source, name=name, job_cls=Source)

    def add_job(self, mapper=None, sources=(), requires=(), name=None,
            job_cls=Job, **kwargs):
        """
        add a job to the gob
        sources are files that will be passed as input to the mapper.
        requires are names of jobs that must run before the mapper.
        """
        if isinstance(sources,basestring):
            sources = (sources,)
        if not name:
            name = mapper.__name__.lower()

        if name in self.jobs:
            raise ValueError('attempt to insert second job with same name')

        source_jobs = [self.jobs[s] for s in sources]
        for req in requires:
            if req not in self.jobs:
                raise LookupError('dependencies must be added first')

        # FIXME: this is UGLY.
        if kwargs.get('saver') == 'split_save':
            split = True
        elif kwargs.get('reducer') or job_cls==Cat:
            split = False
        elif source_jobs:
            split = any(j.split_data for j in source_jobs)
        else:
            split = False

        job = job_cls(mapper=mapper,
                     sources=source_jobs,
                     name=name,
                     split_data=split,
                      **kwargs)

        self.jobs[name] = job
        return job
