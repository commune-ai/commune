import asyncio

import concurrent
import threading
from typing import *

class Task:

    @classmethod
    def wait(cls, futures:list, timeout:int = None, generator:bool=False, return_dict:bool = True) -> list:
        is_singleton = bool(not isinstance(futures, list))

        futures = [futures] if is_singleton else futures
        # if type(futures[0]) in [asyncio.Task, asyncio.Future]:
        #     return cls.gather(futures, timeout=timeout)
            
        if len(futures) == 0:
            return []
        if cls.is_coroutine(futures[0]):
            return cls.gather(futures, timeout=timeout)
        
        future2idx = {future:i for i,future in enumerate(futures)}

        if timeout == None:
            if hasattr(futures[0], 'timeout'):
                timeout = futures[0].timeout
            else:
                timeout = 30
    
        if generator:
            def get_results(futures):
                try: 
                    for future in concurrent.futures.as_completed(futures, timeout=timeout):
                        if return_dict:
                            idx = future2idx[future]
                            yield {'idx': idx, 'result': future.result()}
                        else:
                            yield future.result()
                except Exception as e:
                    cls.print(f'Error: {e}')
                    yield None
                
        else:
            def get_results(futures):
                results = [None]*len(futures)
                try:
                    for future in concurrent.futures.as_completed(futures, timeout=timeout):
                        idx = future2idx[future]
                        results[idx] = future.result()
                        del future2idx[future]
                    if is_singleton: 
                        results = results[0]
                except Exception as e:
                    unfinished_futures = [future for future in futures if future in future2idx]
                    cls.print(f'Error: {e}, {len(unfinished_futures)} unfinished futures with timeout {timeout} seconds')
                return results

        return get_results(futures)



    @classmethod
    def gather(cls,jobs:list, timeout:int = 20, loop=None)-> list:

        if loop == None:
            loop = cls.get_event_loop()

        if not isinstance(jobs, list):
            singleton = True
            jobs = [jobs]
        else:
            singleton = False

        assert isinstance(jobs, list) and len(jobs) > 0, f'Invalid jobs: {jobs}'
        # determine if we are using asyncio or multiprocessing

        # wait until they finish, and if they dont, give them none

        # return the futures that done timeout or not
        async def wait_for(future, timeout):
            try:
                result = await asyncio.wait_for(future, timeout=timeout)
            except asyncio.TimeoutError:
                result = {'error': f'TimeoutError: {timeout} seconds'}

            return result
        
        jobs = [wait_for(job, timeout=timeout) for job in jobs]
        future = asyncio.gather(*jobs)
        results = loop.run_until_complete(future)

        if singleton:
            return results[0]
        return results
    



    @classmethod
    def submit(cls, 
                fn, 
                params = None,
                kwargs: dict = None, 
                args:list = None, 
                timeout:int = 40, 
                return_future:bool=True,
                init_args : list = [],
                init_kwargs:dict= {},
                executor = None,
                module: str = None,
                mode:str='thread',
                max_workers : int = 100,
                ):
        kwargs = {} if kwargs == None else kwargs
        args = [] if args == None else args
        if params != None:
            if isinstance(params, dict):
                kwargs = {**kwargs, **params}
            elif isinstance(params, list):
                args = [*args, *params]
            else:
                raise ValueError('params must be a list or a dictionary')
        
        fn = cls.get_fn(fn)
        executor = cls.executor(max_workers=max_workers, mode=mode) if executor == None else executor
        args = cls.copy(args)
        kwargs = cls.copy(kwargs)
        init_kwargs = cls.copy(init_kwargs)
        init_args = cls.copy(init_args)
        if module == None:
            module = cls
        else:
            module = cls.module(module)
        if isinstance(fn, str):
            method_type = cls.classify_fn(getattr(module, fn))
        elif callable(fn):
            method_type = cls.classify_fn(fn)
        else:
            raise ValueError('fn must be a string or a callable')
        
        if method_type == 'self':
            module = module(*init_args, **init_kwargs)

        future = executor.submit(fn=fn, args=args, kwargs=kwargs, timeout=timeout)

        if not hasattr(cls, 'futures'):
            cls.futures = []
        
        cls.futures.append(future)
            
        
        if return_future:
            return future
        else:
            return cls.wait(future, timeout=timeout)

    @classmethod
    def submit_batch(cls,  fn:str, batch_kwargs: List[Dict[str, Any]], return_future:bool=False, timeout:int=10, module = None,  *args, **kwargs):
        n = len(batch_kwargs)
        module = cls if module == None else module
        executor = cls.executor(max_workers=n)
        futures = [ executor.submit(fn=getattr(module, fn), kwargs=batch_kwargs[i], timeout=timeout) for i in range(n)]
        if return_future:
            return futures
        return cls.wait(futures)

   
    executor_cache = {}
    @classmethod
    def executor(cls, max_workers:int=None, mode:str="thread", cache:bool = True, maxsize=200, **kwargs):
        if cache:
            if mode in cls.executor_cache:
                return cls.executor_cache[mode]
        executor =  cls.module(f'executor.{mode}')(max_workers=max_workers, maxsize=maxsize , **kwargs)
        if cache:
            cls.executor_cache[mode] = executor
        return executor
    


    @staticmethod
    def detailed_error(e) -> dict:
        import traceback
        tb = traceback.extract_tb(e.__traceback__)
        file_name = tb[-1].filename
        line_no = tb[-1].lineno
        line_text = tb[-1].line
        response = {
            'success': False,
            'error': str(e),
            'file_name': file_name,
            'line_no': line_no,
            'line_text': line_text
        }   
        return response
    

    @classmethod
    def as_completed(cls , futures:list, timeout:int=10, **kwargs):
        return concurrent.futures.as_completed(futures, timeout=timeout)
    
    @classmethod
    def is_coroutine(cls, future):
        """
        returns True if future is a coroutine
        """
        return cls.obj2typestr(future) == 'coroutine'


    @classmethod
    def obj2typestr(cls, obj):
        return str(type(obj)).split("'")[1]

    @classmethod
    def tasks(cls, task = None, mode='pm2',**kwargs) -> List[str]:
        kwargs['network'] = 'local'
        kwargs['update'] = False
        modules = cls.servers( **kwargs)
        tasks = getattr(cls, f'{mode}_list')(task)
        tasks = list(filter(lambda x: x not in modules, tasks))
        return tasks


    @classmethod
    def asubmit(cls, fn:str, *args, **kwargs):
        
        async def _asubmit():
            kwargs.update(kwargs.pop('kwargs',{}))
            return fn(*args, **kwargs)
        return _asubmit()
