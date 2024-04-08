
import commune as c


class Wallet(c.Module):


    def __init__(self, network='main', **kwargs):
        self.set_config(locals())

    def call(self, *text, **kwargs):
        return self.talk(*text, **kwargs)

