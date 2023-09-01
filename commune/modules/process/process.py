import commune as c
import multiprocessing as mp
import os
from typing import *
class Process(c.Module):
    
    process_map = {}
    @classmethod
    def add(cls,fn: Union['callable', str],  
                    args:list = None, 
                    kwargs:dict = None, 
                    daemon:bool = True, 
                    tag = None,
                    start:bool = True,
                    tag_seperator:str=':'):

        if isinstance(fn, str):
            fn = c.resolve_fn(fn)
        if args == None:
            args = []
        if kwargs == None:
            kwargs = {}

        assert callable(fn), f'target must be callable, got {fn}'
        assert  isinstance(args, list), f'args must be a list, got {args}'
        assert  isinstance(kwargs, dict), f'kwargs must be a dict, got {kwargs}'
        
        t = processing.process(target=fn, args=args, kwargs=kwargs)
        t.__dict__['start_time'] = c.time()
        t.daemon = daemon

        if start:
            t.start()
        fn_name = fn.__name__
        if tag == None:
            tag = ''
        else:
            tag = str(tag)
        name = fn_name + tag_seperator + tag
        cnt = 0
        while name in cls.process_map:
            cnt += 1
            name = fn_name + tag_seperator + tag + str(cnt)

        cls.process_map[name] = t

        return t

    @classmethod
    def getppid(cls):
        return os.getppid()

    @classmethod
    def getpid(cls):
        return os.getpid()

    def get_age(self, name):
        return c.time() - self.process_map[name].start_time

    def oldest_process_name(self):
        oldest_name = None
        oldest_age = 0
        assert len(self.process_map) > 0, 'no processes to join'
        for name in self.process_map.keys():

            if self.get_age(name) < oldest_age:
                oldest_age = self.get_age(name)
                oldest_name = name
        return oldest_name

    def oldest_process(self):
        oldest_name = self.oldest_process_name()
        oldest_process = self.process_map[oldest_name]
        return oldest_process

    def oldest_pid(self):
        return self.oldest_process().pid

    def join_processes(self, processs:[str, list]):

        processs = self.process_map
        for t in processs:
            self.join_process(t)

    @classmethod
    def fleet(cls, fn:str, n=10,  tag:str=None,  args:list = None, kwargs:dict=None):
        args = args or []
        kwargs = kwargs or {}
        processs = []
        if tag == None:
            tag = ''
        for i in range(n):
            t = cls.process(fn=fn, tag=tag+str(i), *args, **kwargs)
        return cls.process_map


    @classmethod
    def processes(cls, *args, **kwargs):
        return list(cls.process_map(*args, **kwargs).keys())

