
import multiprocessing as mp

class MPQueue:
    def __init__(self,max_workers=2):
        self.process = {}

    def submit(self, fn, args=[], kwargs={}):
        p = mp.Process(target=fn, args=args, kwargs=kwargs)
        self.process[fn.__name__] = p

    def running(self):
        return list(self.process.keys())
    def stop(self, key):
        self.process[key].stop()
        