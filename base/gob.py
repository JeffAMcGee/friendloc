import inspect

class Job(object):
    """
    Abstract base class representing a job to run.
    """
    def __init__(self, source=None, split_input=False, split_output=False,
                 name=None):
        self.name = name or source.__name__.lower()
        self.source = source
        self.split_input = split_input
        self.split_output = split_output

    def run(self):
        if inspect.isclass(self.source):
            obj = self.source()
            func = obj.run
        else:
            func = self.source
        return self.process(func)

    def process(self, func):
        raise NotImplementedError

class Producer(Job):
    def process(self,func):
        # FIXME!!!
        return func()

class Gob(object):
    def __init__(self):
        # Do we want this to be some kind of singleton?
        self.jobs = {}

    def run_job(self,name):
        return self.jobs[name].run()

    def add_job(self, job):
        if job.name in self.jobs:
            raise ValueError('attempt to insert a second job with same name')

        self.jobs[job.name] = job

    def maper(self, *args, **kwargs):
        job = Maper(*args, **kwargs)
        self.add_job(job)
        return job

    def source(self, *args, **kwargs):
        job = Source(*args, **kwargs)
        self.add_job(job)
        return job

    def list_reducer(self, *args, **kwargs):
        job = ListReducer(*args, **kwargs)
        self.add_job(job)
        return job

    def producer(self, *args, **kwargs):
        job = Producer(*args, **kwargs)
        self.add_job(job)
        return job

