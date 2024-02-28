import commune as c

class ValiWatchdog(c.Module):

    def sync_loop(self, remote=None):
        c.thread(c.module('subspace').check_servers)

    