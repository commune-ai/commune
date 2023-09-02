import commune as c
Process = c.module('process')
class ProcessPool(Process):
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

        self.input_queue = Process.queue(maxsize=max_queue_size)
        self.output_queue = Process.queue(maxsize=max_queue_size)

        self.fn = fn
        c.thread(self.output_collector, kwargs=dict(output_queue=self.output_queue))


    def start_workers(self,  num_workers:int = 4):
        for i in range(num_workers):
            self.start(fn=self.run, kwargs=dict(fn=fn, queue = self.input_queue, semaphore=self.semaphore(num_workers), output_queue=self.output_queue), tag=f'worker_{i}')


    def submit(self, 
             fn = None,
             kwargs = None,
             wait_until_response:bool = False, 
             prefix = None,
             timeout = 10):
        if fn == None:
            fn = self.fn
        assert callable(fn), f'fn must be callable, got {fn}'
        if kwargs == None:
            kwargs = {}
        start_time = c.time()

        if self.input_queue.full():
            c.print('Input queue is full, waiting for space')
            while self.input_queue.full():
                c.sleep(0.1)

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
                asyncio.sleep(0.1)
                c.print(self.outputs)
            return self.outputs.pop(kwargs_key)

            return {'status': 'success', 'key': kwargs_key}

    def output_collector(self, output_queue:'mp.Queue',  sleep_time:float=0.0):
        '''
        Collects output from the output_queue and stores it in self.outputs
        '''
        c.print('Starting output collector')
        while True:
            c.sleep(sleep_time)
            
            c.print('output_queue.qsize()', output_queue.qsize())
            output = output_queue.get()
            kwargs_key = output.pop('key')
            result = output['result']
            c.print('kwargs_key', kwargs_key)
            self.outputs[kwargs_key] = result


    def run(self, fn:'callable', queue:'mp.Queue', output_queue:'mp.Queue',  semaphore:'mp.Semaphore'):
        c.new_event_loop()
        while True:
            kwargs = queue.get()
            kwargs_key = kwargs.pop('kwargs_key')
            result = fn(**kwargs)
            self.put_json('results/'+kwargs_key, {'kwargs': kwargs, 'result': result})
            output_queue.put({'key': kwargs_key, 'result': result, 'kwargs': kwargs})




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

        # c.print(self.outputs)
        while self.output_queue.qsize() > 0:
            # c.print(self.output_queue.qsize(), 'outputs remaining')
            # c.print(self.outputs)
            c.sleep(1)



        



        


