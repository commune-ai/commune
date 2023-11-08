import commune as c
import multiprocessing as mp
import os
from typing import *
class Process(c.Module):
    
    process_map = {}
    @classmethod
    def queue(cls, *args, **kwargs):
        return mp.Queue(*args, **kwargs)
    @classmethod
    def process(cls,fn: Union['callable', str],  
                    args:list = None, 
                    kwargs:dict = None, 
                    daemon:bool = True, 
                    tag:str = None,
                    name:str = None,
                    start:bool = True,
                    tag_seperator:str=':'):

        if args == None:
            args = []
        if kwargs == None:
            kwargs = {}
        assert  isinstance(args, list), f'args must be a list, got {args}'
        assert  isinstance(kwargs, dict), f'kwargs must be a dict, got {kwargs}'

        if isinstance(fn, str):
            fn = c.get_fn(fn)
        assert callable(fn), f'target must be callable, got {fn}'
        t = mp.Process(target=fn, args=args, kwargs=kwargs)
        fn_name = fn.__name__

        t.__dict__['start_time'] = c.time()

        t.daemon = daemon
        if start:
            t.start()
        if tag == None:
            tag = ''
        else:
            tag = str(tag)

        if name == None:
            name = fn_name
        name = name + tag_seperator + tag
        cnt = 0
        while name in cls.process_map:
            cnt += 1
            name = fn_name + tag_seperator + tag + str(cnt)

        cls.process_map[name] = t

        return t

    start = process 

    @classmethod
    def getppid(cls):
        return os.getppid()

    @classmethod
    def getpid(cls):
        return os.getpid()

    @classmethod
    def get_age(cls, name):
        return c.time() - cls.process_map[name].start_time

    @classmethod
    def oldest_process_name(cls):
        oldest_name = None
        oldest_age = 0
        assert len(cls.process_map) > 0, 'no processes to join'
        for name in cls.process_map.keys():

            if cls.get_age(name) > oldest_age:
                oldest_age = cls.get_age(name)
                oldest_name = name

        return oldest_name

    @classmethod
    def oldest_process(cls):
        oldest_name = cls.oldest_process_name()
        oldest_process = cls.process_map[oldest_name]
        return oldest_process

    @classmethod
    def oldest_pid(cls):
        return cls.oldest_process().pid

    @classmethod
    def n(cls):
        return len(cls.process_map)


    def join(cls):
        processs = list(cls.process_map.keys())
        for p_name in processs:
            cls.stop(p_name)
        return {'success':True, 'msg':'processes stopped', 'n':cls.n()}

    stop_all = join

    @classmethod
    def stop(cls, name=None):
        if name == None:
            name = cls.oldest_process_name()
        assert name in cls.process_map, f'process {name} not found'
        p = cls.process_map.pop(name)
        p.join()
        assert p.is_alive() == False, f'process {name} is still alive'
        return {'success':True, 'name':name, 'msg':'process removed', 'n':cls.n()}

    @classmethod
    def remove_oldest(cls):
        name = cls.oldest_process_name()
        return cls.remove(name)

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

    @classmethod
    def fn(cls):
        return 1
    

    @classmethod
    def test(cls, n=10):


        for i in range(n):
            cls.start(fn=cls.fn, tag='test', start=True)
            c.print('Started process', i+1, 'of', n, 'processes')
            assert cls.n() == i+1, 'process not added'
        
        for i in range(n):
            cls.stop()
            c.print('Stopped process', i+1, 'of', n, 'processes')
            assert cls.n() == n-i-1, 'process not removed'
        assert cls.n() == 0, 'process not removed'  

        return {'success':True, 'msg':'processes started and stopped'}      

    @classmethod
    def semaphore(cls, n:int = 100):
        semaphore = mp.Semaphore(n)
        return semaphore

    def __delete__(self):
        self.join()
        return {'success':True, 'msg':'processes stopped'}
    

    

