
import commune as c

class ValiWatchdog(c.Module):

    def __init__(self, sleep=60, max_tries=100, num_loops=5):

        self.sleep = sleep
        self.max_tries = max_tries
        for i in range(num_loops):
            c.thread(self.loop)

    def loop(self, cache_exceptions=False):
        if cache_exceptions:
            try:
                self.loop(cache_exceptions=cache_exceptions)
            except Exception as e:

                self.loop(cache_exceptions=cache_exceptions)
            subspace = c.module('subspace')()
            
        while True:
            c.print( f'Checking servers {c.time()}')
            subspace.check_servers()
            c.print('Sleeping {} seconds...')
            c.sleep(self.sleep)


    