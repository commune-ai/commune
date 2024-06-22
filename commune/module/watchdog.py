import commune as c

class Watchdog(c.Module):
    futures = []
    def __init__(self, 
                 modules=['module', 'subspace'], 
                 sleep_time=60, 
                 timeout=10):
        self.modules = {m: c.module(m) for m in modules}
        self.sleep_time = sleep_time
        self.timeout = timeout
        c.thread(self.run_loop)
    def sync(self, blocking=False):
        if not blocking:
             self.futures.append(c.submit(self.sync))
        self.modules = {m: c.module(m) for m in self.modules}
        c.print('syncing...')
        c.ip(update=1)
        c.tree(update=1)
        c.namespace(update=1)
        c.print('synced')
        self.subspace = c.module('subspace')
        self.subspace.balances(update=1)
        self.subspace.stake_from(update=1)
        self.subspace.namespace(update=1)

    def run_loop(self):
        while True:
            try:
                self.futures  = [f for f in self.futures if not f.done()]
                self.futures.append(c.submit(self.sync))
            except Exception as e:
                e = c.detailed_error(e)
                c.print(f'Error syncing, restarting {e}')
                self.restart_self()
            
            c.sleep(self.sleep_time)

                
