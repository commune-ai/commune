import commune as c

class Test(c.Module):
    @classmethod
    def test(cls):
        self = c.module('subspace.test')
        n = self.n()
        assert isinstance(n, int)
        assert n > 0

        market_cap = self.mcap()
        assert isinstance(market_cap, int), f'{market_cap} is not an int'

        name2key = self.name2key()
        assert isinstance(name2key, dict), f'{name2key} is not a dict'
        assert len(name2key) == n

        stats = self.stats()
        assert len(stats) == n

        assert isinstance(self.names()[0], str)
        assert isinstance(self.addresses()[0], str)

        assert isinstance(self.global
                          _params(), dict)
        assert isinstance(self.params

        



    def test_global_params(self):
        global_params = c.global_params()


        


