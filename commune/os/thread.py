

    
import threading

from typing import *
import commune as c

class Thread(c.Module):
    thread_map = {}
    @classmethod
    def thread(cls,fn: Union['callable', str],  
                    args:list = None, 
                    kwargs:dict = None, 
                    daemon:bool = True, 
                    name = None,
                    tag = None,
                    start:bool = True,
                    tag_seperator:str='::', 
                    **extra_kwargs):
        
        if isinstance(fn, str):
            fn = c.get_fn(fn)
        if args == None:
            args = []
        if kwargs == None:
            kwargs = {}

        assert callable(fn), f'target must be callable, got {fn}'
        assert  isinstance(args, list), f'args must be a list, got {args}'
        assert  isinstance(kwargs, dict), f'kwargs must be a dict, got {kwargs}'
        
        # unique thread name
        if name == None:
            name = fn.__name__
            cnt = 0
            while name in cls.thread_map:
                cnt += 1
                if tag == None:
                    tag = ''
                name = name + tag_seperator + tag + str(cnt)
        
        if name in cls.thread_map:
            cls.thread_map[name].join()

        t = threading.Thread(target=fn, args=args, kwargs=kwargs, **extra_kwargs)
        # set the time it starts
        setattr(t, 'start_time', c.time())
        t.daemon = daemon
        if start:
            t.start()
        cls.thread_map[name] = t
        return t

    @classmethod
    def join_threads(cls, threads:[str, list]):

        threads = cls.thread_map
        for t in threads.values():
            # throw error if thread is not in thread_map
            t.join()
        return {'success': True, 'msg': 'all threads joined', 'threads': threads}

    @classmethod
    def threads(cls, search:str=None, **kwargs):
        threads = list(cls.thread_map.keys())
        if search != None:
            threads = [t for t in threads if search in t]
        return threads

    @classmethod
    def thread_count(cls):
        return threading.active_count()
    
    
    