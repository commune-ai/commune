import commune as c

class ValiParity(c.Module):
    def __init__(self, config = None, **kwargs):
        self.set_config(config, kwargs=kwargs)
        self.subspace = c.module('subspace')()
        self.subnet = self.subspace.subnet()

    def votes(self, max_trust = 25) -> int:
        modules = self.subspace.modules()
        voted_modules = c.shuffle([m for m in modules if m['trust'] < max_trust])[:self.subnet['max_allowed_weights']]
        uids = [m['uid'] for m in voted_modules]
        weights = [m['trust'] for m in voted_modules]
        return {'uids': uids, 'weights': weights}
    

    def run(self, config = None, **kwargs):
        while True:
            votes = self.votes()
            c.vote(**votes, key=self.key)
            self.sleep(60)

    