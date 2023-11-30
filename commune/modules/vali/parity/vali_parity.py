import commune as c

class ValiParity(c.Module):
    def __init__(self, run=True):
        self.subspace = c.module('subspace')()
        self.subnet = self.subspace.subnet()
        if run:
            c.thread(self.run)
        self.seconds_per_epoch = self.subnet['tempo'] * 8

    def votes(self, max_trust = 25) -> int:
        modules = self.subspace.modules()
        voted_modules = c.shuffle([m for m in modules if m['trust'] < max_trust])[:self.subnet['max_allowed_weights']]
        uids = [m['uid'] for m in voted_modules]
        weights = [m['trust'] for m in voted_modules]
        return {'uids': uids, 'weights': weights}
    

    def run(self):
        while True:
            c.print('voting...')
            r = self.vote()
            c.print(r)
            self.sleep(self.seconds_per_epoch)

    def vote(self):
        try:
            votes = self.votes()
            response = self.subspace.vote(**votes, key=self.key)
        except Exception as e:
            e = c.detailed_error(e)
            c.print(e)
        return response

    