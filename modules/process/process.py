import commune as c
import multiprocessing as mp
import os
from typing import *
class Process(c.Module):
    process_map = {}
    def queue(self, *args, **kwargs):
        return mp.Queue(*args, **kwargs)
    def resolve_function(self, fn):
        if isinstance(fn, str):
            fn = c.get_fn(fn)
        assert callable(fn), f'target must be callable, got {fn}'
        return fn
    


    def process(self,fn: Union['callable', str],  
                    args:list = None, 
                    kwargs:dict = None, 
                    daemon:bool = True, 
                    tag:str = None,
                    name:str = None,
                    start:bool = True,
                    tag_seperator:str='::'): 
        args = args or []
        kwargs = kwargs or {}
        fn = self.resolve_function(fn)
        t = mp.Process(target=fn, args=args, kwargs=kwargs)
        fn_name = fn.__name__
        t.__dict__['start_time'] = c.time()
        t.daemon = daemon
        if start:
            t.start()
        name = name or fn_name 
        if tag != None:
            name = name + tag_seperator + tag
        
        name = self.ensure_unique_name(name)
        self.process_map[name] = t
        return t

    start = process 

    def ensure_unique_name(self, name):
        cnt = 0
        new_name = name + str(cnt)
        while new_name in self.process_map:
            cnt += 1
            new_name = name + str(cnt)
        return new_name
    
    def getppid(self):
        return os.getppid()

    
    def getpid(self):
        return os.getpid()

    
    def get_age(self, name):
        return c.time() - self.process_map[name].start_time

    
    def oldest_process_name(self):
        oldest_name = None
        oldest_age = 0
        assert len(self.process_map) > 0, 'no processes to join'
        for name in self.process_map.keys():

            if self.get_age(name) > oldest_age:
                oldest_age = self.get_age(name)
                oldest_name = name

        return oldest_name

    
    def oldest_process(self):
        oldest_name = self.oldest_process_name()
        oldest_process = self.process_map[oldest_name]
        return oldest_process

    
    def oldest_pid(self):
        return self.oldest_process().pid

    
    def n(self):
        return len(self.process_map)


    def join(self):
        processs = list(self.process_map.keys())
        for p_name in processs:
            self.stop(p_name)
        return {'success':True, 'msg':'processes stopped', 'n':self.n()}

    stop_all = join

    
    def stop(self, name=None):
        if name == None:
            name = self.oldest_process_name()
        assert name in self.process_map, f'process {name} not found'
        p = self.process_map.pop(name)
        p.join()
        assert p.is_alive() == False, f'process {name} is still alive'
        return {'success':True, 'name':name, 'msg':'process removed', 'n':self.n()}

    
    def remove_oldest(self):
        name = self.oldest_process_name()
        return self.remove(name)

    
    def fleet(self, fn:str, n=10,  tag:str=None,  args:list = None, kwargs:dict=None):
        args = args or []
        kwargs = kwargs or {}
        processs = []
        if tag == None:
            tag = ''
        for i in range(n):
            t = self.process(fn=fn, tag=tag+str(i), *args, **kwargs)
        return self.process_map


    
    def processes(self, *args, **kwargs):
        return list(self.process_map(*args, **kwargs).keys())

    
    def fn(self):
        return 1
    

    
    def test(self, n=10):


        for i in range(n):
            self.start(fn=self.fn, tag='test', start=True)
            c.print('Started process', i+1, 'of', n, 'processes')
            assert self.n() == i+1, 'process not added'
        
        for i in range(n):
            self.stop()
            c.print('Stopped process', i+1, 'of', n, 'processes')
            assert self.n() == n-i-1, 'process not removed'
        assert self.n() == 0, 'process not removed'  

        return {'success':True, 'msg':'processes started and stopped'}      

    
    def semaphore(self, n:int = 100):
        semaphore = mp.Semaphore(n)
        return semaphore

    def __delete__(self):
        self.join()
        return {'success':True, 'msg':'processes stopped'}
    

    

