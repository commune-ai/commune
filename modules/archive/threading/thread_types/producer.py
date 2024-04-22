
import threading

class ProducerThread(threading.Thread):
    r""" This producer thread runs in backgraound to fill the queue with the result of the target function.
    """
    def __init__(self, target, arg=[], kwargs={}, name=None):
        r"""Initialization.
        Args:
            queue (:obj:`queue.Queue`, `required`)
                The queue to be filled.
                
            target (:obj:`function`, `required`)
                The target function to run when the queue is not full.

            arg (:type:`tuple`, `required`)
                The arguments to be passed to the target function.

            name (:type:`str`, `optional`)
                The name of this threading object. 
        """
        super(ProducerThread,self).__init__()
        self.name = name
        self.target = target
        self.arg = arg
        self.queue = queue 
        self._stop_event = threading.Event()

    def run(self):
        r""" Work of the thread. Keep checking if the queue is full, if it is not full, run the target function to fill the queue.
        """
        while not self.stopped():
            if not self.queue.full():
                item = self.target(*self.arg, self.queue.qsize()+1 )
                self.queue.put(item)
            time.sleep(2)
        return

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()
