import commune as c

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
            fn = c.resolve_fn(fn)
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

    def join_threads(self, threads:[str, list]):

        threads = self.thread_map
        for t in threads:
            self.join_thread(t)

    @classmethod
    def thread_fleet(cls, fn:str, n=10,  tag:str=None,  args:list = None, kwargs:dict=None):
        args = args or []
        kwargs = kwargs or {}
        threads = []
        if tag == None:
            tag = ''
        for i in range(n):
            t = cls.thread(fn=fn, tag=tag+str(i), *args, **kwargs)
        return cls.thread_map


    @classmethod
    def threads(cls, *args, **kwargs):
        return list(cls.thread_map(*args, **kwargs).keys())

