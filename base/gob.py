from __future__ import print_function

import contextlib
import inspect
import itertools
import logging
import sys
import sqlite3
import os.path
import traceback
from collections import defaultdict
from multiprocessing import Pool
import signal
import json
import re
import pickle
import functools

import msgpack

try:
    import numpy
except ImportError:
    pass


class StatefulIter(object):
    "wrap an iterator and keep the most recent value in self.current"
    NOT_STARTED = object()
    FINISHED = object()

    def __init__(self,it):
        self.it = iter(it)
        self.current = self.NOT_STARTED
        self.index = -1

    def __iter__(self):
        return self

    def next(self):
        try:
            self.current = next(self.it)
            self.index+=1
        except StopIteration:
            self.current = self.FINISHED
            self.index = None
            raise
        if self.index and self.index%1000==0:
            logging.info("Iter at %d, current is %r",self.index,self.current)
        return self.current


def mapper(all_items=False,merge=None,slurp=None):
    def wrapper(f):
        f.all_items = all_items
        f.merge = merge
        f.slurp = slurp or {}
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
def avg_reduce(key, items, rereduce):
    # avg is stored as sum and count so that rereduce works
    items = tuple(items)
    if rereduce:
        return sum(it[0] for it in items),len(items)
    else:
        return sum(items),len(items)


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
    def __init__(self, source_names=(), requires=(), saver='save', name=None,
                 split_data="UNKNOWN"):
        self.source_names = source_names
        self.requires = tuple(source_names)+tuple(requires)
        self.name = name
        self.split_data = split_data
        self.sources = None # this will be set when it is added to a gob

    def added(self, gob):
        """This is called when a Job is added to a gob."""
        pass

    def output_files(self, env, reduced=True):
        if self.split_data or (not reduced and self.split_sources()):
            # FIXME: why were we changing split_data here?
            #self.split_data = any(j.split_data for j in self.sources)
            # FIXME: hasattr is ugly
            if hasattr(self,'encoding'):
                return list(env.split_files(self.name, self.encoding))
            else:
                return list(env.split_files(self.name))
        return [self.name,]

    def split_sources(self):
        return any(j.split_data for j in self.sources)


class MapJob(Job):
    def __init__(self, mapper=None, saver='save', reducer=None,
                 procs=6, encoding='mp', **kwargs):

        super(MapJob,self).__init__(**kwargs)
        self.mapper = mapper
        self.saver = saver
        self.reducer = reducer
        self.procs = procs
        self.encoding = encoding

    def added(self, gob):
        # set split_data
        if self.saver == 'split_save':
            self.split_data = True
        elif self.reducer:
            self.split_data = False
        elif self.sources:
            self.split_data = self.split_sources()
        else:
            self.split_data = False

    def output_name(self, in_paths):
        "the output name for a set of inputs"
        # FIXME: how do we want to handle split and unsplit inputs?
        # FIXME: this is ugly.
        suffix = ''
        for path in in_paths:
            name = os.path.basename(path)
            pos = name.find('.')
            if pos==-1:
                continue
            if suffix:
                assert suffix==name[pos:]
            else:
                suffix = name[pos:]

        return self.name + suffix

    def load_output(self, path, env):
        "load the results of a previous run of this job"
        return env.load(path, encoding=self.encoding)


class Cat(Job):
    def __init__(self, pattern=None, **kwargs):
        super(Cat,self).__init__(split_data=False, **kwargs)
        if isinstance(pattern,basestring):
            self.pattern = re.compile(pattern)
        else:
            self.pattern = pattern

    def load_output(self, path, env):
        job = self.sources[0]
        files = job.output_files(env)
        if self.pattern:
            files = filter(self.pattern.match,files)
        loads = (job.load_output(p,env) for p in files)
        return itertools.chain.from_iterable(loads)


class Source(Job):
    def __init__(self, source_func=None, **kwargs):
        super(Source,self).__init__(split_data=False, **kwargs)
        self.source_func = source_func

    def load_output(self,path,env):
        return self.source_func()


class Clump(Job):
    def __init__(self, clump_func, clumps, **kwargs):
        super(Clump,self).__init__(split_data=True, **kwargs)
        self.clump_func = clump_func
        self.clumps = clumps

    def output_files(self, env):
        return ["%s.%s"%(self.name,clump) for clump in self.clumps]

    def load_output(self,path,env):
        parent_job = self.sources[0]
        clump = path.rsplit('.',1)[1]
        parent_files = parent_job.output_files(env)
        files = [ f
                for f in parent_files
                if self.clump_func(f.split('.')[1:],clump)
                ]

        loads = (parent_job.load_output(p,env) for p in files)
        return itertools.chain.from_iterable(loads)


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


    def run(self, job, input_paths, slurped):
        """
        run the job. save the results. block until it is done.

        Subclasses of this should behave like a drop-in replacement for
        SingleThreadExecutor.run(...) .
        """
        raise NotImplementedError

    def map_reduce_save(self, job, in_paths, slurped):
        "completely process one file -> map, reduce, and save"
        assert len(job.sources) == len(in_paths)

        output = job.output_name(in_paths)
        if (
                self.job_status(output)=='done' and
                self.name_exists(output, encoding=job.encoding)
            ):
            logging.info("skipping map for %s - already done",output)
            return
        self.set_job_status(output,'started')

        inputs = [
                src.load_output(path, self)
                for src, path in zip(job.sources,in_paths)
                ]
        results = self.map_single(job.mapper,inputs,in_paths,slurped)
        if job.reducer:
            results = self.reduce_single(job.reducer,results)

        self.save_single(job, in_paths, results)
        self.set_job_status(output,'done')

    def map_single(self, mapper, inputs, in_paths, slurped):
        "run mapper for one set of inputs and return an iterator of results"
        all_items = mapper.all_items or not inputs
        if all_items:
            sfi_inputs = [StatefulIter(it) for it in inputs]
            item_iters = [ sfi_inputs ]
        else:
            item_iters = itertools.izip_longest(*inputs)

        for args in item_iters:
            try:
                results = _call_opt_kwargs(
                            mapper,
                            *args,
                            in_paths=in_paths,
                            env=self,
                            **slurped)
                if results:
                    for res in results:
                        yield res
            except:
                if all_items:
                    current_args = [it.current for it in sfi_inputs]
                else:
                    current_args = args
                self._handle_map_err(mapper,current_args)
                raise

    def _handle_map_err(self, mapper, args):
        msg = "map %r failed for %r"%(mapper,args)
        if self.log_crashes:
            with open('gobstop.%s.%d'%(mapper.__name__,os.getpid()),'w') as gs:
                print(msg,file=gs)
                traceback.print_exc(file=gs)


    def reduce_single(self, reducer, map_output):
        grouped = defaultdict(list)
        for k,v in map_output:
            grouped[k].append(v)
        return [
            ( key, _call_opt_kwargs(reducer,key,items,rereduce=False) )
            for key,items in grouped.iteritems()
            ]

    def save_single(self, job, in_paths, results):
        if job.saver:
            saver = getattr(self,job.saver)
            saver(job.output_name(in_paths),results,encoding=job.encoding)
        else:
            # force the generator to generate
            for res in results:
                pass

    def split_save(self, name, key_item_pairs, encoding=None):
        with self.bulk_saver(name,encoding) as saver:
            for key,value in key_item_pairs:
                saver.add(_path(name,key),value)

    def reduce_all(self, job):
        if not job.reducer:
            return
        grouped = defaultdict(list)
        for path in self.split_files(job.name,job.encoding):
            for k,v in self.load(path):
                grouped[k].append(v)
        results = [
            ( key, _call_opt_kwargs(job.reducer,key,items,rereduce=True) )
            for key,items in grouped.iteritems()
            ]
        self.save(job.name,results,encoding=job.encoding)


class Storage(object):
    """
    AbstractBaseClass for Storage objects.

    Subclasses of this class must implement all the methods that raise a
    NotImplementedError.
    """
    STATUSES = ["new","started","done","crashed"]

    def name_exists(self, name, encoding=None):
        """
        True if there is a file for name, False otherwise.
        If this returns false, self.load(name) should raise an exception.
        """
        raise NotImplementedError

    def save(self, name, items, encoding=None):
        "store items in the file 'name'. items is an iterator."
        raise NotImplementedError

    def load(self, name, encoding=None):
        "return an iterator over the items stored in the file 'name'"
        raise NotImplementedError

    def bulk_saver(self, name, encoding=None):
        """
        context manager that returns a BulkSaver object, which can save to
        multiple open files at the same time. The object will have this method:
            def add(self, name, item)
        When that method is called, it will append item to the file with the
        name 'name'.
        """
        raise NotImplementedError

    def split_files(self, name, encoding=None):
        "return an iterator of the names that name was split into"
        raise NotImplementedError

    def input_paths(self, source_jobs):
        # FIXME: look at merge here
        inputs = [sorted(job.output_files(self)) for job in source_jobs]
        if not all(inputs):
            raise ValueError("missing dependencies for job")
        if not inputs:
            return [[]]
        if len(inputs)>=2:
            first_len = len(inputs[0])
            # FIXME: check that the files actually match?
            if not all(len(inp)==first_len for inp in inputs[1:]):
                raise ValueError("job with multiple sources and uneven files")

        return zip(*inputs)

    def job_status(self, name):
        "return the status field for an input"
        raise NotImplementedError

    def set_job_status(self, name, status):
        "change the status field for an input"
        raise NotImplementedError


class SingleThreadExecutor(Executor):
    "Execute in a single process"
    def run(self, job, input_paths, slurped):
        """
        This is the cannonical implementation of run for an Executor.
        Subclasses of executor should be roughly equivalent to this method, but
        they can run map_reduce_save in different processes or on different
        machines.
        """
        for source_set in input_paths:
            self.map_reduce_save(job, source_set, slurped)
        self.reduce_all(job)

    def _handle_map_err(self, mapper, args):
        msg = "map %r failed for %r"%(mapper,args)
        logging.exception(msg)
        print(msg)
        traceback.print_exc()
        import ipdb
        ipdb.post_mortem(sys.exc_info()[2])


class DictStorage(Storage):
    "Store data in a dict in this class for testing and debugging."
    THE_FS = {}
    JOB_DB = {}

    def save(self, name, items, encoding=None):
        self.THE_FS[name] = list(items)

    def load(self, name, encoding=None):
        return iter(self.THE_FS[name])

    def name_exists(self, name, encoding=None):
        return name in self.THE_FS

    class BulkSaver(object):
        def __init__(self):
            self.data = defaultdict(list)

        def add(self, name, item):
            self.data[name].append(item)

    @contextlib.contextmanager
    def bulk_saver(self, name, encoding=None):
        bs = DictStorage.BulkSaver()
        yield bs
        for key,items in bs.data.iteritems():
            self.THE_FS[key] = items

    def split_files(self, name, encoding=None):
        return (n for n in self.THE_FS if n.startswith(name+'.'))

    def job_status(self, name):
        return self.JOB_DB.get(name,{}).get('status','new')

    def set_job_status(self, name, status):
        state = self.JOB_DB.setdefault(name,{})
        state['status'] = status


class FileStorage(Storage):
    "Store data in a directory"
    def __init__(self, path, **kwargs):
        super(FileStorage,self).__init__(**kwargs)
        # path should be an absolute path to a directory to store files
        self.path = os.path.abspath(path)
        self._job_db = None
        self._job_pid = None

        with self._cursor(commit=True) as cur:
            cur.execute("""
                create table if not exists outputs
                ( name text primary key, status text )
                """)

    def _path(self, name, encoding):
        parts = name.split('.')
        return os.path.join(self.path,*parts)+'.'+encoding

    def _open(self, name, encoding, mode='rb'):
        path = self._path(name,encoding)
        dir = os.path.dirname(path)
        if 'w' in mode:
            try:
                os.makedirs(dir)
            except OSError:
                # attempt to make it and then check to avoid race condition
                if not os.path.exists(dir):
                    raise
        return open(path,mode)

    def name_exists(self, name, encoding='mp'):
        return os.path.exists(self._path(name,encoding))

    def _encoder_ending(self,encoding):
        if encoding =='mp':
            packer = msgpack.Packer()
            return packer.pack,''
        elif encoding =='json':
            return json.dumps,'\n'
        elif encoding =='pkl':
            return functools.partial(pickle.dumps,protocol=2),''
        else:
            raise ValueError('unsupported encoding')

    def save(self, name, items, encoding='mp'):
        with self._open(name,encoding,'wb') as f:
            if encoding=='npz':
                # this obviously fails if numpy is missing
                numpy.savez(f,*list(items))
            else:
                encoder,ending = self._encoder_ending(encoding)
                for item in items:
                    print(encoder(item),end=ending,file=f)

    def load(self, name, encoding='mp'):
        # Can we just return the iterator or is close a problem?
        with self._open(name,encoding) as f:
            if encoding =='mp':
                for item in msgpack.Unpacker(f,encoding='utf-8'):
                    yield item
            elif encoding =='json':
                for line in f:
                    yield json.loads(line)
            elif encoding =='npz':
                # this obviously fails if numpy is missing
                npzfile = numpy.load(f)
                for key,item in npzfile.iteritems():
                    yield item
            elif encoding =='pkl':
                uper = pickle.Unpickler(f)
                try:
                    while True:
                        yield uper.load()
                except EOFError:
                    return
            else:
                raise ValueError('unsupported encoding')


    class BulkSaver(object):
        def __init__(self,storage,encoding):
            self.files = {}
            self.encoding = encoding
            self.encoder,self.ending = storage._encoder_ending(encoding)
            self.storage = storage

        def add(self, name, item):
            if name not in self.files:
                self.files[name] = self.storage._open(name,self.encoding,'w')
            print(self.encoder(item),end=self.ending,file=self.files[name])

        def _close(self):
            for file in self.files.itervalues():
                file.close()

    @contextlib.contextmanager
    def bulk_saver(self, name, encoding='mp'):
        bs = self.BulkSaver(self, encoding)
        yield bs
        bs._close()

    def split_files(self, name, encoding='mp'):
        suffix = '.'+encoding
        for dirpath, dirs, files in os.walk(os.path.join(self.path,name)):
            _dir = dirpath.replace(self.path,'').lstrip('/').replace('/','.')
            for file in files:
                if file.endswith(suffix):
                    yield "%s.%s"%(_dir,file[:-len(suffix)])

    @contextlib.contextmanager
    def _cursor(self,commit=False):
        pid = os.getpid()
        if not self._job_db or self._job_db_pid!=pid:
            self._job_db_pid = pid
            self._job_db = sqlite3.connect(
                                os.path.join(self.path,'job.db'),
                                timeout=10)
        cursor = self._job_db.cursor()
        yield cursor
        if commit:
            self._job_db.commit()
        cursor.close()


    def job_status(self, name):
        with self._cursor() as cur:
            cur.execute( 'select status from outputs where name=?', (name,) )
            row = cur.fetchone()
        return row[0] if row else 'new'

    def set_job_status(self, name, status):
        with self._cursor(commit=True) as cur:
            cur.execute(
                "replace into outputs (name, status) values ( ?, ? )",
                (name, status)
            )


def _usr1_handler(sig, frame):
    traceback.print_stack(frame)
    print(frame.f_locals)


def _mp_worker_init(env, job, slurped):
    MultiProcEnv._worker_data['env'] = env
    MultiProcEnv._worker_data['job'] = job
    MultiProcEnv._worker_data['slurped'] = slurped
    env.setup_logging('%s.%d'%(job.name,os.getpid()))
    signal.signal(signal.SIGUSR1, _usr1_handler)


def _mp_worker_run(source_set):
    # The map method must be a top-level method, not a class method because
    # pickle is broken.
    env = MultiProcEnv._worker_data['env']
    job = MultiProcEnv._worker_data['job']
    slurped = MultiProcEnv._worker_data['slurped']
    # letting exceptions bubble outside of the process confuses Pool
    try:
        env.map_reduce_save(job, source_set, slurped)
    except:
        logging.exception("crash is child proc")
        return False
    return True


class MultiProcEnv(FileStorage, Executor):
    "Use python's multiprocessing to split up a job"

    _worker_data = {}

    def run(self, job, input_paths, slurped):
        pool = Pool(job.procs,_mp_worker_init,[self, job, slurped])
        results = pool.map(_mp_worker_run, input_paths, chunksize=1)
        pool.close()
        pool.join()

        if not all(results):
            raise Exception("child job failed!")

        self.reduce_all(job)
        MultiProcEnv._worker_data = {}


class SimpleEnv(DictStorage,SingleThreadExecutor):
    "Run in a single thread, store results in ram"


class SimpleFileEnv(FileStorage,SingleThreadExecutor):
    "Run in a single thread, store results to disk"


class Gob(object):
    def __init__(self, env):
        # Is env really different from gob? How?
        # Do we want this to be some kind of singleton?
        self.jobs = {}
        self.env = env

    def clear_job(self,name):
        job = self.jobs[name]
        for f in job.output_files(self.env,reduced=False):
            # FIXME: should this delete the files?
            self.env.set_job_status(f,'new')

    def run_job(self,name):
        job = self.jobs[name]
        slurped = self.slurp(job.mapper.slurp)
        input_paths = self.env.input_paths(job.sources)
        return self.env.run(job,input_paths=input_paths,slurped=slurped)

    def add(self, job):
        """ add a job to this gob """
        if job.name in self.jobs:
            raise ValueError('attempt to insert second proc with same name')

        for req in job.requires:
            if req not in self.jobs:
                raise LookupError('dependencies must be added first')

        self.jobs[job.name] = job
        job.sources = [self.jobs[s] for s in job.source_names]
        job.added(self)
        if job.split_data=="UNKNOWN":
            raise ValueError('split_data must be set in __init__ or added')
        return job

    def add_cat(self, name, source, pattern=None):
        """ concatenate a split source into one source """
        return self.add(Cat(source_names=(source,), name=name, pattern=pattern))

    def add_clump(self, clump_func, clumps, source, name=None):
        """
        Adds a clump that takes several split sources and combines them into a
        different set of split sources.
        """
        if not name:
            name = clump_func.__name__.lower()
        clump = Clump(clump_func=clump_func,
                      clumps=clumps,
                      source_names=(source,),
                      name=name)
        return self.add(clump)

    def add_source(self, source, name=None):
        """
        Adds a source that the you, the library user, create. This can be used
        to provide starting data.
        """
        if not name:
            name = source.__name__.lower()
        return self.add(Source(source_func=source, name=name))

    def add_map_job(self, mapper=None, source_names=(), name=None, **kwargs):
        """
        add a job to the gob
        sources are files that will be passed as input to the mapper.
        """
        if isinstance(source_names,basestring):
            source_names = (source_names,)
        if not name:
            name = mapper.__name__.lower()

        job = MapJob(
                mapper=mapper,
                source_names=source_names,
                name=name,
                **kwargs)

        return self.add(job)

    def _cat_job_output(self,key):
        job = self.jobs[key]
        files = job.output_files(self.env)
        loads = (job.load_output(p,self.env) for p in files)
        return itertools.chain.from_iterable(loads)

    def slurp(self,converters):
        return {
            key:converter(self._cat_job_output(key))
            for key,converter in converters.iteritems()
        }
