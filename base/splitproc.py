#!/usr/bin/env python
import pdb
import logging
import sys, os
import multiprocessing
import Queue


BUFFER_DEFAULT = 1000


class SplitProcess(object):
    """
    Take a set of tasks and split it up to several processors.

    To use SplitProcess, make a subclass and override produce and map.  Then,
    create an instance and call run().  Run effectively does this:
        return self.consume(self.map(self.produce()))

    Caveats:
    * Subclasses of SplitProcess must be pickleable.
    * The order that results get to consume is undefined.
    * map() must return one and only one item for every item it is given.
    * I am worried that under certain conditions not all of the results
      get passed to consume.  You could make self._done a JoinableQueue .
    """
    def __init__(self, label="proc", slaves=8, buffer_size=0, 
            log_path="logs", log_level=None, debug=False, *args, **kwargs):
        self.slave_id = -1
        self.halting = multiprocessing.Event()
        self.debug = debug
        self._proc_label = label
        self._log_path = log_path
        self._log_level = log_level
        self._slaves = slaves
        self._args = args
        self._kwargs = kwargs

        self._file_hdlr = None
        buffer_size = buffer_size or BUFFER_DEFAULT
        self._todo = multiprocessing.JoinableQueue(buffer_size)
        self._done = multiprocessing.Queue()

    def startup(self):
        "this is called when the master or slave starts running"
        pass

    def produce(self):
        """generator that returns a stream of objects to act on"""
        raise NotImplementedError

    def map(self,items):
        """generator that processes items from produce"""
        raise NotImplementedError

    def consume(self,items):
        """Process each of the results from map.  Items is an iterator. """
        for item in items:
            pass
        
    def run(self):
        if self.debug:
            return self.run_single()
        slaves = self._start_slaves()
        self._setup_logging(self._proc_label)
        self.startup(*self._args,**self._kwargs)

        jobs = self.produce()
        result = self.consume(self._do_async(jobs))
        for slave in slaves:
            slave.join()
        self._stop_logging()
        return result

    def _start_slaves(self):
        procs =[]
        for slave_id in xrange(self._slaves):
            p = multiprocessing.Process(
                target = SplitProcess._run_slave,
                args = [self,slave_id],
            )
            p.start()
            procs.append(p)
        return procs

    def _run_slave(self,slave_id):
        #when this method starts, we are running in a new process!
        self.slave_id = slave_id
        self._setup_logging("%s_%d"%(self._proc_label,slave_id))
        try:
            logging.warn("thread %d began",slave_id)
            self.startup(*self._args,**self._kwargs)
            for result in self.map(self._todo_iter()):
                if result is not None:
                    logging.debug("slave %d wrote %r",slave_id,result)
                self._done.put(result)
                self._todo.task_done()
        except:
            logging.exception("exception killed thread")
            self._todo.task_done()
        self._stop_logging()

    def _todo_iter(self):
        "generator that gets items from self.todo"
        while True:
            try:
                task = self._todo.get(True,.1)
                logging.debug("slave %d read %r",self.slave_id,task)
                yield task
            except Queue.Empty:
                if not self.halting.is_set():
                    continue
                logging.warn("thread %d finished",self.slave_id)
                return

    def _setup_logging(self, label):
        if self._log_level is None:
            return
        root = logging.getLogger()
        if len(root.handlers)==0:
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
            self.file_hdlr = file_hdlr

    def _stop_logging(self):
        if self._file_hdlr:
            logging.getLogger().removeHandler(self.file_hdlr)

    def _do_async(self,jobs):
        "put jobs on self._todo and yield results from self._done"
        logging.warn("started producing jobs")
        running = True
        while running:
            try:
                job = jobs.next()
                self._todo.put(job)
            except StopIteration:
                running = False
                self.halting.set()
                logging.warn("finished producing jobs")
                self._todo.join()
                logging.warn("all slaves are done")
            while not self._done.empty():
                try:
                    yield self._done.get_nowait()
                except Queue.Empty:
                    pass
 
    def run_single(self):
        "run the job without multiprocessing for testing and debuging"
        self._setup_logging('single')
        try:
            self.startup(*self._args,**self._kwargs)
            res = self.consume(self.map(self.produce()))
        except:
            logging.exception("exception caused HALT")
            pdb.post_mortem()
        self._stop_logging()
        return res
