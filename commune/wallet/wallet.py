import commune as c

class Wallet(c.Module):

    _fns = ['registered_servers', 
            'transfer', 
            'transfer_stake',
            'stake_transfer',
            'stake',
            'set_weights',
            'get_weights',
            'module_info',
            'stats',
            'get_balance',
            'get_stake',
            'get_profit_shares',
            'add_profit_shares',
            'send']

    def __init__(self):
        self.sync()

    def registered_servers(self, *args, **kwargs):
        return self.registered_servers(*args, **kwargs)

    def sync(self):
        self.subspace = c.module('subspace')()
        for fn in self._fns:
            setattr(self, fn, getattr(self.subspace, fn))
