#!/usr/bin/env python
import pdb
import logging
import sys, os
import multiprocessing
import Queue


BUFFER_DEFAULT = 1000


class SplitProcess(object):
    def __init__(self, label="proc", slaves=8, buffer_size=0, 
            log_path="logs", log_level=None):
        self.slave_id = -1
        self.halting = multiprocessing.Event()
        self._proc_label = label
        self._log_path = log_path
        self._log_level = log_level
        self._slaves = slaves

        buffer_size = buffer_size or BUFFER_DEFAULT
        self._todo = multiprocessing.JoinableQueue(buffer_size)
        self._done = multiprocessing.Queue()

    def produce(self):
        """generator that returns a stream of objects to act on"""
        raise NotImplementedError

    def map(self,item):
        """process one item and return a result"""
        raise NotImplementedError

    def consume(self,items):
        """Process each of the results from map.  Items is an iterator. """
        pass
        
    def run_procs(self):
        slaves = self._start_slaves()
        self._setup_logging(self._proc_label)
        jobs = self.produce()
        result = self.consume(self._do_async(jobs))
        for slave in slaves:
            slave.join()
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
            while True:
                try:
                    task = self._todo.get(True,.1)
                    logging.debug("slave %d read %r",slave_id,task)
                except Queue.Empty:
                    if not self.halting.is_set():
                        continue
                    logging.warn("thread %d finished",slave_id)
                    return
                result = self.map(task)
                if result is not None:
                    logging.debug("slave %d wrote %r",slave_id,result)
                self._done.put(result)
                self._todo.task_done()
        except:
            logging.exception("exception killed thread")
            self._todo.task_done()

    def _setup_logging(self, label):
        if self._log_level is None:
            return
        root = logging.getLogger()
        if self._log_path is not None:
            filepath = os.path.join(self._log_path, label)
            file_hdlr = logging.FileHandler(filepath, 'a')
            fmt = logging.Formatter(logging.BASIC_FORMAT, None)
            file_hdlr.setFormatter(fmt)
            root.addHandler(file_hdlr)
            file_hdlr.setLevel(self._log_level)
        hdlr = logging.StreamHandler()
        root.addHandler(hdlr)
        hdlr.setLevel(self._log_level)
        root.setLevel(logging.DEBUG)

    def _do_async(self,jobs):
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
        self._setup_logging('single')
        try:
            return self.consume(self.map(d) for d in self.produce())
        except:
            logging.exception("exception caused HALT")
            pdb.post_mortem()
