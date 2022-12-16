

from munch import Munch 
class AyncioManager:
    """ Base threadpool executor with a priority queue 
    """

    def __init__(self,  max_tasks:int=10):
        """Initializes a new ThreadPoolExecutor instance.
        Args:
            max_threads: 
                The maximum number of threads that can be used to
                execute the given calls.
        """
        self.max_tasks = max_tasks
        self.running, self.stopped = False, False
        self.tasks = []
        self.queue = Munch({'in':queue.Queue(), 'out':queue.Queue()})
        self.start()

    def stop(self):
        while self.running:
            self.stopped = True
        return self.stopped
        
    def start(self):
        self.background_thread = threading.Thread(target=self.run_loop, args={}, kwargs={}, daemon=True)
        self.background_thread.start()

    def run_loop(self):
        return asyncio.run(self.async_run_loop())
    def new_aysnc_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop
    async def async_run_loop(self): 
        loop = self.new_aysnc_loop()
        print(loop)
        self.stopped = False
        self.running = True
        print(loop)

        while self.running and not self.stopped:
            finished_tasks = []
            if len(self.tasks)>0:
                finished_tasks, self.tasks = await asyncio.wait(self.tasks)
            for finished_task in finished_tasks:
                self.queue.out.put(await asyncio.gather(*finished_task))
            if len(self.tasks) <= self.max_tasks:
                new_job = self.queue['in'].get()
                self.submit(**new_job)
                new_job = self.queue.out.get()

        loop.close()
        self.running = False

    def submit(self,fn, *args, **kwargs):
        job = {'fn': fn, 'args': args, 'kwargs': kwargs}
        self.queue['in'].put(job)

    def get(self):
        return self.queue['out'].get()

    def close(self):
        for task in self.tasks:
            task.cancel()
        self.stop()
        self.background_thread.join()

    def __del__(self):
        self.close()
