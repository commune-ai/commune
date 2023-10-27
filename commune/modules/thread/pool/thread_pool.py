import commune as c
Thread = c.module('thread')
import asyncio
from typing import *
import gc
class ThreadPool(Thread):
    def __init__(self, 
                fn : Optional['Callable'] = None,
                 num_workers:int = 4, 
                 max_queue_size:int = 100, 
                 verbose: bool = False, 
                 save_outputs : bool= False,
                 path = None):
                

        self.path = path if path != None else self.resolve_path('store')
        self.verbose = verbose
        self.input_queue = self.queue(maxsize=max_queue_size)
        self.output_queue = self.queue(maxsize=max_queue_size)
        self.save_outputs = save_outputs

        # start workers
        for i in range(num_workers):
            kwargs =dict(fn=fn, queue = self.input_queue, semaphore=self.semaphore(num_workers),output_queue=self.output_queue)
            self.thread(fn=self.run, kwargs=kwargs , tag=f'worker_{i}')

        self.fn = fn

        # run output collector as a thread to collect the output from the output_queue
        c.thread(self.output_collector, kwargs=dict(output_queue=self.output_queue))




    reserverd_keys = ['kwargs_key']

    def submit(self, 
             fn = None,
             kwargs : dict = None,
             wait_until_response:bool = False, 
             timeout = 10, 
             sleep_inteval = 0.01):
        start_time = c.time()
        if kwargs == None:
            kwargs = {}

        assert all([key not in kwargs for key in self.reserverd_keys]), f'{self.reserverd_keys} are reserved keys'
        kwargs_key = c.hash(kwargs) +'_T'+str(c.time())

        input = {'kwargs_key': kwargs_key, 'kwargs': kwargs, 'fn': fn}
        # if self.verbose:
        self.input_queue.put(input)

        if wait_until_response:
            c.print('Waiting for response')
            start_time = c.time()
            while kwargs_key not in self.outputs:
                asyncio.sleep(sleep_inteval)
                if c.time() - start_time > timeout:
                    raise TimeoutError(f'Timeout after {timeout} seconds')
            return self.outputs.pop(kwargs_key)

            return {'status': 'success', 'key': kwargs_key}

    def output_collector(self, output_queue:'mp.Queue',  sleep_time:float=0.0):
        '''
        Collects output from the output_queue and stores it in self.outputs
        '''
        while True:
            c.sleep(sleep_time)
            output = output_queue.get()
            if self.save_outputs:
                self.save_output(output)

    def save_output(self, output):
        kwargs_key = output.pop('key')
        output_path = self.resolve_output_path(kwargs_key)
        self.put(output_path, output)

    def get_output(self, kwargs_key):
        output_path = self.resolve_output_path(kwargs_key)
        return self.get(output_path)

    def resolve_output_path(self, kwargs_key):
        return f'{self.tag}/'+kwargs_key

    def ls_outputs(self):
        return self.ls(f'{self.tag}')
    def output_exists(self, kwargs_key):
        return self.exists(self.resolve_output_path(kwargs_key))

    def time2output(self):
        outputs = self.ls_outputs()
        timestamp2output = {int(output.split('_T')[-1].split('.')[0]): output for output in outputs}
        return timestamp2output

    def oldest_output(self):
        timestamp2output = self.time2output()
        oldest_timestamp = min(timestamp2output.keys())
        return timestamp2output[oldest_timestamp]

    def run(self, fn:'callable', queue:'mp.Queue', output_queue:'mp.Queue',  semaphore:'mp.Semaphore'):
        c.new_event_loop()
        while True:

            try:
                input = queue.get()
                kwargs = input['kwargs']
                kwargs_key = input['kwargs_key']
                if input['fn'] != None:
                    tmp_fn = input['fn']
                    assert callable(tmp_fn), f'fn must be callable, got {tmp_fn}'
                    result = tmp_fn(**kwargs)
                    fn_name = tmp_fn.__name__
                else:
                    result = fn(**kwargs)
                    fn_name = fn.__name__
                
                output_queue.put({'key': kwargs_key, 'result': result, 'kwargs': kwargs, 'time': c.time(), 'fn': fn_name})
                ## remove memory
                del kwargs
                del result
                del kwargs_key
                del fn_name
                del input
                # garbage collect
                gc.collect()
            except Exception as e:
                c.print({'status': 'error', 'error': e})



    def is_full(self):
        return self.input_queue.full()


    @property
    def num_tasks(self):
        return self.input_queue.qsize()





    
    @classmethod
    def test(cls):
        def fn(x):
            result =  x*2
            return result
            
        self = cls(fn=fn)
        for i in range(100):
            self.submit(kwargs=dict(x=i))

        while self.num_tasks > 0:
            c.print(self.num_tasks)


        return {'success': True, 'msg': 'thread pool test passed'}

        


