import commune as c

Process = c.module('process')
class ProcessPool(Process):
    def __init__(self, 
                fn = None,
                 max_workers:int = 10, 
                 max_queue_size:int = 100, 
                 verbose: bool = False, 
                 path = None):
                
        self.inputs = {}
        self.outputs = {}

        self.path = path if path != None else self.resolve_path('store')

        self.verbose = verbose

        self.queue = Process.queue(maxsize=max_queue_size)
        self.output_queue = Process.queue(maxsize=max_queue_size)
        self.fn = fn
        c.thread(self.output_collector, kwargs=dict(output_queue=self.output_queue))

        for i in range(max_workers):
            self.start(fn=self.run, kwargs=dict(fn=fn, queue = self.queue, semaphore=self.semaphore(max_workers), output_queue=self.output_queue), tag=f'worker_{i}')
    def submit(self, **kwargs:dict):
        assert 'kwargs_key' not in kwargs, 'kwargs_key is a reserved key'
        kwargs_key = c.hash(kwargs)
        kwargs['kwargs_key'] = kwargs_key
        self.queue.put(kwargs)
        kwargs_key = c.hash(kwargs) +'_T'+str(c.timestamp())
        self.inputs[kwargs_key] = kwargs
        return {'status': 'success', 'kwargs_key': kwargs_key}

    def output_collector(self, output_queue:'mp.Queue',  sleep_time:float=0.0):
        c.print('Starting output collector')
        while True:
            c.sleep(sleep_time)
            c.print('Output collector sleeping', self.output_queue.qsize())
            output = output_queue.get()
            kwargs_key = output.pop('key')
            result = output['result']
            c.print('Output collector received output for', kwargs_key)
            c.print('Output collector received output for', kwargs_key)
            self.outputs[kwargs_key] = result


    def run(self, fn:'callable', queue:'mp.Queue', output_queue:'mp.Queue',  semaphore:'mp.Semaphore'):

        while True:
            kwargs = queue.get()
            kwargs_key = kwargs.pop('kwargs_key')
            result = fn(**kwargs)
            # self.put_json(kwargs_key, {'kwargs': kwargs, 'result': result})
            output_queue.put({'key': kwargs_key, 'result': result, 'kwargs': kwargs})





    def num_tasks(self):
        return self.queue.qsize()


    
    @classmethod
    def test(cls):
        def fn(x):
            result =  x*2
            return result
            
        self = cls(fn=fn)
        for i in range(100):
            self.submit(x=i)

        while self.output_queue.qsize() > 0:
            # c.print(self.num_tasks(), 'tasks remaining')
            c.print(self.output_queue.qsize(), 'outputs remaining')
            # c.print(self.output_queue.get())
            # time.sleep(1)

        c.print('Finished submitting tasks')
        c.print(self.outputs)
        # while len(self.outputs) < 100:
        #     c.print('Waiting for outputs')
        #     # time.sleep(1)



        



        


