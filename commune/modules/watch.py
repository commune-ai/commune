
import commune as c

class Watch(c.Module):

    def __init__(self, sync_period=60):
        self.sync_period = sync_period
        c.print('ValiWatchdog initialized')
        c.thread(self.run_loop)
    def run_loop(self):
        while True:
            try:
                self.sync()
                c.print('ValiWatchdog: Updated, sleeping for', self.sync_period)
            except Exception as e:
                c.print(e)
            c.sleep(self.sync_period)

    def sync(self):
        c.ip(update=1)
        c.namespace(update=1)



