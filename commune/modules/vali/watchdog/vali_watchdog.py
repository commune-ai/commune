
import commune as c

class ValiWatchdog(c.Module):

    def __init__(self, sleep=60, max_tries=100):

        self.sleep = sleep
        self.max_tries = max_tries
        c.thread(self.loop)



    def loop(self):
        try:
            subspace = c.module('subspace')()
            while True:
                c.print( f'Checking servers {c.time()}')
                c.thread(c.module('subspace')().check_servers)
                c.print('Sleeping {} seconds...')
                c.sleep(self.sleep)
        except Exception as e:
            c.print(f'Error in watchdog: {e}, retrying.. {self.max_tries} tries left.')
            self.max_tries = self.max_tries - 1
            assert self.max_tries > 0, 'Max tries reached'
            self.loop()

    