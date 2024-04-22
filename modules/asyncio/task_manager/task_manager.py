
import concurrent
import asyncio

_shutdown = False

def execute_hello(ind):
    if _shutdown:
        print(f"Thread {threading.current_thread().name}: Skipping task {ind} as shutdown was requested")
        return None
    ...


class AsyncioThreadExecutor:

    def __init__(self, max_threads=10):
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)


    async def run(self, fn:callable, tasks:list):

        loop = asyncio.get_running_loop()
        futures =[]

        for task in tasks:
            task_args = list(task) if type(task) in [list, set, tuple] else []
            task_kwargs = dict(task) if type(task) == dict else {}
            futures.append(loop.run_in_executor(self.thread_pool, fn, **task_kwargs))
            
        try:
            results = await asyncio.gather(*futures, return_exceptions=False)
        except Exception as ex:
            print("Caught error executing task", ex)
            _shutdown = True
            raise
        return results