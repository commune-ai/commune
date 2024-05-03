
import commune as c

class ValiWatchdog(c.Module):

    def __init__(self, sleep=60, max_tries=100, search='vali', cache_exceptions=False):
        # self.init_module(locals())
        c.print('ValiWatchdog initialized')
        c.thread(self.run_loop)
    def run_loop(self):
        subspace = c.module('subspace')()
        while True:
            try:
                c.print( f'Checking servers {c.time()}')
                c.print(subspace.check_servers('vali'))
                c.print('Sleeping {} seconds...')
                c.sleep(self.config.sleep)
            except Exception as e:
                c.print(f'Error: {e}')
                c.sleep(self.config.sleep)
                if not self.config.cache_exceptions:
                    raise e
            


    