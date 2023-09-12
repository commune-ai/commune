import commune as c
Thread = c.module('thread')
import asyncio
class ThreadPool(Thread):
    def __init__(self, 
                fn = None,
                 num_workers:int = 4, 
                 max_queue_size:int = 100, 
                 verbose: bool = False, 
                 path = None):
                
        self.inputs = {}
        self.outputs = {}

        self.path = path if path != None else self.resolve_path('store')

        self.verbose = verbose

        self.input_queue = self.queue(maxsize=max_queue_size)
        self.output_queue = self.queue(maxsize=max_queue_size)


        for i in range(num_workers):
            self.thread(fn=self.run, kwargs=dict(fn=fn, queue = self.input_queue, semaphore=self.semaphore(num_workers), output_queue=self.output_queue), tag=f'worker_{i}')

        self.fn = fn
        c.thread(self.output_collector, kwargs=dict(output_queue=self.output_queue))




    def submit(self, 
             fn = None,
             kwargs= None,
             wait_until_response:bool = False, 
             prefix = None,
             timeout = 10, 
             sleep_inteval = 0.01):
        start_time = c.time()
        if kwargs == None:
            kwargs = {}
        assert 'kwargs_key' not in kwargs, 'kwargs_key is a reserved key'

        kwargs_key = c.hash(kwargs) +'_T'+str(c.time())
        if prefix != None:
            kwargs_key = prefix + '_' + kwargs_key

        input = {'kwargs_key': kwargs_key, 'kwargs': kwargs, 'fn': fn}
        kwargs['kwargs_key'] = kwargs_key


        self.input_queue.put(kwargs)
        self.inputs[kwargs_key] = kwargs

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
            kwargs_key = output.pop('key')
            result = output['result']
            self.outputs[kwargs_key] = result


    def run(self, fn:'callable', queue:'mp.Queue', output_queue:'mp.Queue',  semaphore:'mp.Semaphore'):
        c.new_event_loop()
        while True:
            kwargs = queue.get()
            kwargs_key = kwargs.pop('kwargs_key')
            if 'fn' in kwargs:
                tmp_fn = kwargs.pop('fn')
                tmp_fn(**kwargs)
            result = fn(**kwargs)
            output_queue.put({'key': kwargs_key, 'result': result, 'kwargs': kwargs, 'time': c.time()})
            ## remove memory
            del kwargs
            del result
            del kwargs_key



    def is_full(self):
        return self.input_queue.full()


    @property
    def num_tasks(self):
        return self.input_queue.qsize()



    def __del__(self):
        self.close()
        


    
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



        



        


