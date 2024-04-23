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


        



    def test_global_params(self):
        global_params = c.global_params(fmt='dict')
        assert isinstance(global_params, dict)




    def test_subnet_params(self):
        subnet_params_a = c.subnet_params(netuid=0)
        assert isinstance(subnet_params_a, dict)
        subnet_params_b = c.subnet_params(netuid='commune')
        assert isinstance(subnet_params_b, dict)
        assert c.hash(subnet_params_a) == c.hash(subnet_params_b), f'{subnet_params_a} != {subnet_params_b}'

        subnet_params_all = c.subnet_params(netuid='all')
        assert isinstance(subnet_params_all, dict)
        for k,v in subnet_params_all.items():
            assert isinstance(v, dict), f'{v} is not a dict'

        return {'msg': 'subnet_params test passed', 'success': True}






        


