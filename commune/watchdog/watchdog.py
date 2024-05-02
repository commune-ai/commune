import commune as c

class Watchdog(c.Module):
    def __init__(self):
        c.thread(self.run_loop)

    def sync(self):
        c.print('syncing...')
        c.ip(update=1)
        c.tree(update=1)
        c.namespace(update=1)
        c.print('synced')
    def run_loop(self, sleep_time=60):
        while True:
            try:
                self.sync()
            except Exception as e:
                print(e)
                pass
            c.sleep(sleep_time)
            print(f'sleeping for {sleep_time} seconds')
