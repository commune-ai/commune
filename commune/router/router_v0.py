import commune as c
import asyncio
import gc



import os
import sys
import time
import queue
import random
import weakref
import itertools
import threading

from loguru import logger
from typing import Callable
import concurrent
from concurrent.futures._base import Future
import commune as c
import gc

Task = c.module('router.task')

NULL_ENTRY = (sys.maxsize, Task.null())

class Router(c.Module):
    """Base threadpool executor with a priority queue"""

    # Used to assign unique thread names when thread_name_prefix is not supplied.
    _counter = itertools.count().__next__
    # submit.__doc__ = _base.Executor.submit.__doc__
    threads_queues = weakref.WeakKeyDictionary()

    def __init__(
        self,
        max_workers: int =None,
        maxsize : int =-1,
        thread_name_prefix : str ="",
    ):
        """Initializes a new ThreadPoolExecutor instance.
        Args:
            max_workers: The maximum number of threads that can be used to
                execute the given calls.
            thread_name_prefix: An optional name prefix to give our threads.
        """

        max_workers = (os.cpu_count() or 1) * 5 if max_workers == None else max_workers
        if max_workers <= 0:
            raise ValueError("max_workers must be greater than 0")
            
        self.max_workers = max_workers
        self.task_queue = queue.PriorityQueue(maxsize=maxsize)
        self.idle_semaphore = threading.Semaphore(0)
        self.threads = []
        self.broken = False
        self.shutdown = False
        self.shutdown_lock = threading.Lock()
        self.thread_name_prefix = thread_name_prefix or ("ThreadPoolExecutor-%d" % self._counter() )

    @property
    def is_empty(self):
        return self.task_queue.empty()
    
    def task_path(self, status='pending', tag=None):
        tag= tag or self.tag or "base"
        path = self.resolve_path(tag) + '/' + status
        return path


    def tasks(self, status='pending', tag=None):
        return self.ls(self.task_path(status=status, tag=tag))
    def completed(self,  tag=None):
        return self.ls(self.task_path(status='complete', tag=tag))
    
    def failed(self,  tag=None):
        return self.ls(self.task_path(status='failed', tag=tag))
    
    def refresh_tasks(self, tag=None):
        tag= tag or self.tag or "base"
        task_path = self.resolve_path(tag)
        return c.rm(task_path)

        
    
    def pending(self,  tag=None):
        return self.ls(self.task_path(status='pending', tag=tag))
    

    def submit(self,
                module: str = 'module',
                fn : str = 'info',
                args:dict=None, 
                kwargs:dict=None, 
                
                ):
        return self.call(module=module, fn=fn, args=args, kwargs=kwargs, return_future=True)

    def call(self, 
                module: str = 'module',
                fn : str = 'info',
                args:dict=None, 
                kwargs:dict=None, 
                timeout=200, 
                return_future:bool=False,
                namespace = None,
                network:str='local',
                init_kwargs = None,
                update=False, 
                path:str=None, 
                fn_seperator:str='/',
                priority=1,
                tag = None

                ) -> Future:
        
        args = args or []
        kwargs = kwargs or {}

        tag = tag or self.tag or "base"
        path = self.resolve_path(tag) 

        with self.shutdown_lock:
            if self.broken:
                raise Exception("ThreadPoolExecutor is broken")
            if self.shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")

            task = Task(module=module, 
                                 fn=fn, 
                                 args=args,
                                 kwargs=kwargs, 
                                 timeout=timeout,
                                 path=path, 
                                 priority=priority, 
                                 fn_seperator=fn_seperator, 
                                 namespace=namespace, 
                                 network=network, 
                                 init_kwargs=init_kwargs, 
                                 update=update)
            
            # add the work item to the queue
            self.task_queue.put((task.priority, task), block=False)
            # adjust the thread count to match the new task
            self.adjust_thread_count()
        k = c.timestamp()
        self.tasks[k] = task
        return {'ticket': k}
    

    futures = {}

    def adjust_thread_count(self):
        # if idle threads are available, don't spin new threads
        if self.idle_semaphore.acquire(timeout=0):
            return

        # When the executor gets lost, the weakref callback will wake up
        # the worker threads.
        def weakref_cb(_, q=self.task_queue):
            q.put(NULL_ENTRY)

        num_threads = len(self.threads)
        if num_threads < self.max_workers:
            thread_name = "%s_%d" % (self.thread_name_prefix or self, num_threads)
            t = threading.Thread(
                name=thread_name,
                target=self.worker,
                args=(
                    weakref.ref(self, weakref_cb),
                    self.task_queue,
                ),
            )
            t.daemon = True
            t.start()
            self.threads.append(t)
            self.threads_queues[t] = self.task_queue

    def shutdown(self, wait=True):
        with self.shutdown_lock:
            self.shutdown = True
            self.task_queue.put(NULL_ENTRY)
        if wait:
            for t in self.threads:
                try:
                    t.join(timeout=2)
                except Exception:
                    pass

    @staticmethod
    def worker(executor_reference, task_queue):
        try:
            while True:
                work_item = task_queue.get(block=True)
                priority = work_item[0]

                if priority == sys.maxsize:
                    # Wake up queue management thread.
                    task_queue.put(NULL_ENTRY)
                    break

                item = work_item[1]

                if item is not None:
                    item.run()
                    # Delete references to object. See issue16284
                    del item
                    continue

                executor = executor_reference()
                # Exit if:
                #   - The interpreter is shutting down OR
                #   - The executor that owns the worker has been collected OR
                #   - The executor that owns the worker has been shutdown.
                if executor is None or executor.shutdown:
                    # Flag the executor as shutting down as early as possible if it
                    # is not gc-ed yet.
                    if executor is not None:
                        executor.shutdown = True
                    # Notice other workers
                    task_queue.put(NULL_ENTRY)
                    return
                del executor
        except Exception as e:
            c.print(e, color='red')
            c.print("work_item", work_item, color='red')

            e = c.detailed_error(e)
            c.print("Exception in worker", e, color='red')

    @property
    def num_tasks(self):
        return self.task_queue.qsize()

    @classmethod
    def as_completed(futures: list):
        assert isinstance(futures, list), "futures must be a list"
        return [f for f in futures if not f.done()]

    @staticmethod
    def wait(futures:list) -> list:
        futures = [futures] if not isinstance(futures, list) else futures
        results = []
        for future in c.as_completed(futures):
            results += [future.result()]
        return results

    @classmethod
    def test(cls, tag=None):
        test_module_name = 'test_module'
        module = c.serve(server_name=test_module_name)
        output =  cls().call(module=test_module_name, fn='info', tag=tag)
        c.print(output)
        assert isinstance( output, dict) and 'name' in output 
        assert output['name'] == test_module_name
        c.kill(test_module_name)
        return {'success': True, 'msg': 'thread pool test passed'}


    @classmethod
    def dashboard(cls):
        import streamlit as st
        st.write('ROUTER')
        self = cls()
        self.network = st.selectbox('network', ['local', 'remote', 'subspace', 'bittensor'])
        self.namespace = c.namespace(network=self.network)
        self.playground_dashboard(network=self.network)

    def playground_dashboard(self, network=None, server=None):
        # c.nest_asyncio()
        import streamlit as st

        if network == None:
            network = st.selectbox('network', ['local', 'remote'], 0, key='playground.net')
        else:
            network = network
            namespace = self.namespace
        servers = list(namespace.keys())
        if server == None:
            server_name = st.selectbox('Select Server',servers, 0, key=f'serve.module.playground')
            server = c.connect(server_name, network=network)
        server_info = server.info()
        server_schema = server_info['schema']
        server_functions = list(server_schema.keys())
        server_address = server_info['address']

        fn = st.selectbox('Select Function', server_functions, 0)

        fn_path = f'{self.server_name}/{fn}'
        st.write(f'**address** {server_address}')
        with st.expander(f'{fn_path} playground', expanded=True):

            kwargs = self.function2streamlit(fn=fn, fn_schema=server_schema[fn], salt='sidebar')

            cols = st.columns([3,1])
            timeout = cols[1].number_input('Timeout', 1, 100, 10, 1, key=f'timeout.{fn_path}')
            cols[0].write('\n')
            cols[0].write('\n')
        call = st.button(f'Call {fn_path}')
        if call:
            success = False
            latency = 0
            try:
                t1 = c.time()
                response = getattr(server, fn)(**kwargs, timeout=timeout)
                t2 = c.time()
                latency = t2 - t1
                success = True
            except Exception as e:
                e = c.detailed_error(e)
                response = {'success': False, 'message': e}
            emoji = '✅' if success else '❌'
            latency = str(latency).split('.')[0] + '.'+str(latency).split('.')[1][:2]
            st.write(f'Reponse Status ({latency}s) : {emoji}')
            st.code(response)
    
    @classmethod
    def function2streamlit(cls, 
                           module = None,
                           fn:str = '__init__',
                           fn_schema = None, 
                           extra_defaults:dict=None,
                           cols:list=None,
                           skip_keys = ['self', 'cls'],
                           salt = None,
                            mode = 'pm2'):
        import streamlit as st
        
        key_prefix = f'{module}.{fn}'
        if salt != None:
            key_prefix = f'{key_prefix}.{salt}'
        if module == None:
            module = cls
            
        elif isinstance(module, str):
            module = c.module(module)
        extra_defaults = {} if extra_defaults is None else extra_defaults
        kwargs = {}

        if fn_schema == None:

            fn_schema = module.schema(defaults=True, include_parents=True)[fn]
            if fn == '__init__':
                config = module.config(to_munch=False)
                extra_defaults = config
            fn_schema['default'].pop('self', None)
            fn_schema['default'].pop('cls', None)
            fn_schema['default'].update(extra_defaults)
            fn_schema['default'].pop('config', None)
            fn_schema['default'].pop('kwargs', None)
            
        fn_schema['input'].update({k:str(type(v)).split("'")[1] for k,v in extra_defaults.items()})
        if cols == None:
            cols = [1 for i in list(range(int(len(fn_schema['input'])**0.5)))]
        if len(cols) == 0:
            return kwargs
        cols = st.columns(cols)

        for i, (k,v) in enumerate(fn_schema['default'].items()):
            
            optional = fn_schema['default'][k] != 'NA'
            fn_key = k 
            if fn_key in skip_keys:
                continue
            if k in fn_schema['input']:
                k_type = fn_schema['input'][k]
                if 'Munch' in k_type or 'Dict' in k_type:
                    k_type = 'Dict'
                if k_type.startswith('typing'):
                    k_type = k_type.split('.')[-1]
                fn_key = f'**{k} ({k_type}){"" if optional else "(REQUIRED)"}**'
            col_idx  = i 
            if k in ['kwargs', 'args'] and v == 'NA':
                continue
            

            col_idx = col_idx % (len(cols))
            if type(v) in [float, int] or c.is_int(v):
                kwargs[k] = cols[col_idx].number_input(fn_key, v, key=f'{key_prefix}.{k}')
            elif v in ['True', 'False']:
                kwargs[k] = cols[col_idx].checkbox(fn_key, v, key=f'{key_prefix}.{k}')
            else:
                kwargs[k] = cols[col_idx].text_input(fn_key, v, key=f'{key_prefix}.{k}')
        kwargs = cls.process_kwargs(kwargs, fn_schema)       
        
        return kwargs
    
    
Router.run(__name__)