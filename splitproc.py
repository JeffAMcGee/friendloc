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
        self.slave_id = None
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
        self._start_slaves()
        self._setup_logging(self._proc_label)
        try:
            jobs = self.produce()
            return self.consume(self._do_async(jobs))
        except:
            logging.exception("exception caused HALT")

    def _start_slaves(self):
        for slave_id in xrange(self._slaves):
            p = multiprocessing.Process(
                target = SplitProcess._run_slave,
                args = [self,slave_id],
            )
            p.start()

    def _run_slave(self,slave_id):
        #when this method starts, we are running in a new process!
        self.slave_id = slave_id
        self._setup_logging("%s_%d"%(self._proc_label,slave_id))
        try:
            logging.warn("thread %d began",slave_id)
            while True:
                try:
                    task = self._todo.get(True,1)
                except Queue.Empty:
                    if not self.halting.is_set():
                        continue
                    logging.warn("thread %d finished",slave_id)
                    return
                self._done.put(self.map(task))
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
            file_hdlr.setLevel(logging.INFO)
        hdlr = logging.StreamHandler()
        root.addHandler(hdlr)
        hdlr.setLevel(logging.WARNING)

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
        return self.consume(self.map(d) for d in self.produce())

class TestSplitProc(SplitProcess):
    def produce(self):
        import time
        for x in xrange(10):
            time.sleep(3)
            yield x

    def map(self,item):
        return item+1

    def consume(self,items):
        return sum(items)


if __name__ == '__main__':
    tsp = TestSplitProc(slaves=2)
    print tsp.run_single()
    tsp._log_level=logging.INFO
    print tsp.run_procs()
