import commune as c
import os
import concurrent
import time
class Executor(c.Module):
    def __init__(self, max_workers= None):
        if max_workers is None:
            max_workers = os.cpu_count()
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
        self.task_map = {}


    def task_done(self, future):
        
        future_id = self.future2id(future)
        try:
            result = future.result()
        except Exception as e:
            result = c.detailed_exception(e)
        
        info = future.info
        info['result'] = result
        info['period'] = c.time() - info['time']
        self.task_map.pop(future_id)
        return info

    
    def submit(self, fn, *args, **kwargs):
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
        fn_name = fn.__name__
        future =  self.executor.submit(fn, *args, **kwargs)
        future.add_done_callback(self.task_done)

        # regster future
        future_id = self.future2id(future)
        info = {'fn': fn_name, 'args': args, 'kwargs': kwargs, 'time': time.time(), 'future': future}
        setattr(future, 'info', info)
        self.task_map[future_id] = future


        return future

    @staticmethod
    def future2id(future: concurrent.futures.Future) -> str:
        return hex(id(future))

    @property
    def num_tasks(self):
        return len(self.task_map)

    def shutdown(self):
        self.executor.shutdown(wait=True)
        self.executor = None
        return {'success': True, 'message': 'Executor shutdown'}

    @classmethod
    def test(cls):
        executor = cls()
        def square_number(x):
            return x ** 2
        futures = [executor.submit(square_number, x) for x in range(10)]
        for future in concurrent.futures.as_completed(futures):
            print(future.result())
        executor.shutdown()


