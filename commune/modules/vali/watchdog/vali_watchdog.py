
import commune as c

class ValiWatchdog(c.Module):

    def __init__(self, sleep=60, max_tries=100, num_loops=5):

        self.sleep = sleep
        self.max_tries = max_tries
        for i in range(num_loops):
            c.thread(self.loop)
    def loop(self, cache_exceptions=False):
        subspace = c.module('subspace')()
            
        while True:
            try:
                c.print( f'Checking servers {c.time()}')
                subspace.check_servers()
                c.print('Sleeping {} seconds...')
                c.sleep(self.sleep)
            except Exception as e:
                c.print(f'Error: {e}')
                c.sleep(self.sleep)
                if not cache_exceptions:
                    raise e
            


    