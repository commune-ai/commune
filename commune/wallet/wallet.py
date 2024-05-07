import commune as c

class Wallet(c.Module):

    mirror_fns = ['modules']

    def __init__(self, network='subspace', **kwargs):
        self.subspace = c.module(network)(network=network,  **kwargs)
        for fn in self.mirror_fns:
            setattr(self, fn, getattr(self.subspace, fn))

        