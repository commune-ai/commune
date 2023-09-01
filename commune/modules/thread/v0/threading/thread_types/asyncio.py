
class AsyncioThread(threading.Thread):
    r""" This producer thread runs in backgraound to fill the queue with the result of the target function.
    """
    def __init__(self, fn, arg=[], kwargs={}, name=None, deamon=True, loop=None):
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
        threading.Thread.__init__(self, deamon=deamon, name=name)
        self.name = name
        self.fn = fn
        self.arg = arg
        self._stop_event = threading.Event()
        

    def run(self):

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        item = loop.run_until_complete(self.fn(*self.arg, *self.kwargs) )
        return item

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()
