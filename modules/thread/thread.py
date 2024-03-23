import commune as c
from typing import Union

class Thread(c.Module):
    
    thread_map = {}
    @classmethod
    def thread(cls,fn: Union['callable', str],  
                    args:list = None, 
                    kwargs:dict = None, 
                    daemon:bool = True, 
                    tag = None,
                    start:bool = True,
                    tag_seperator:str=':'):

        if isinstance(fn, str):
            fn = c.get_fn(fn)
        if args == None:
            args = []
        if kwargs == None:
            kwargs = {}

        assert callable(fn), f'target must be callable, got {fn}'
        assert  isinstance(args, list), f'args must be a list, got {args}'
        assert  isinstance(kwargs, dict), f'kwargs must be a dict, got {kwargs}'
        
        import threading
        t = threading.Thread(target=fn, args=args, kwargs=kwargs)
        t.__dict__['time'] = c.time()
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
        while name in cls.thread_map:
            cnt += 1
            name = fn_name + tag_seperator + tag + str(cnt)

        cls.thread_map[name] = t

        return t

    @classmethod
    def queue(cls, maxsize:int = 0):
        import queue
        return queue.Queue(maxsize=maxsize)

    @classmethod
    def join_threads(cls, threads:[str, list]):

        threads = cls.thread_map
        for t in threads.values():
            # throw error if thread is not in thread_map
            t.join()


    @classmethod
    def threads(cls, *args, **kwargs):
        return list(cls.thread_map(*args, **kwargs).keys())

    @classmethod
    def num_threads(cls) -> int:
        return len(cls.thread_map)
    @classmethod
    def test(cls):
        self = cls()
        def fn():
            print('fn')
            c.sleep(1)
            print('fn done')
        self.thread(fn=fn, tag='test')
        c.sleep(2)
        print('done')

    def semaphore(self, n:int = 10):
        import threading
        return threading.Semaphore(n)

