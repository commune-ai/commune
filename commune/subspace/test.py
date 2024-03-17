import commune as c

class Test(c.Module):
    @classmethod
    def test(cls):
        s = c.module('subspace.test')
        n = s.n()
        assert isinstance(n, int)
        assert n > 0

        market_cap = s.mcap()
        assert isinstance(market_cap, int), market_cap

        name2key = s.name2key()
        assert isinstance(name2key, dict)
        assert len(name2key) == n

        stats = s.stats()
        assert len(stats) == n


