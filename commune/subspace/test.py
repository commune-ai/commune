import commune as c

class Test(c.Module):

    def test_global_params(self):
        global_params = c.global_params(fmt='dict')
        assert isinstance(global_params, dict)

    def test_subnet_params(self, netuid=0):
        subnet_params_a = c.subnet_params(netuid=0)
        assert isinstance(subnet_params_a, dict)
        subnet_params_b = c.subnet_params(netuid='commune')
        assert isinstance(subnet_params_b, dict)
        assert c.hash(subnet_params_a) == c.hash(subnet_params_b), f'{subnet_params_a} != {subnet_params_b}'
        subnet_names = c.subnet_names()
        assert isinstance(subnet_names, list) and len(subnet_names) > 0
        subnet2netuid = c.subnet2netuid()
        assert isinstance(subnet2netuid, dict) and len(subnet2netuid) > 0
        namespace = self.namespace(netuid=netuid)
        return {'msg': 'subnet_params test passed', 'success': True}
    


    def test_module_params(self, keys=['dividends', 'incentive'], netuid=0):
        self = c.module('subspace')()
        key = self.keys(update=1, netuid=netuid)[0]
        module_info = self.get_module(key, netuid=netuid)
        assert isinstance(module_info, dict)
        for k in keys:
            assert k in module_info, f'{k} not in {module_info}'

        return {'msg': 'module_params test passed', 'success': True, 'module_info': module_info}


    def test_substrate(self):
        for i in range(3):
            t1 = c.time()
            c.print(c.module('subspace')().substrate)
            t2 = c.time()
            c.print(f'{t2-t1:.2f} seconds')


    def test_global_storage(self):
        global_params = self.global_params(fmt='j')
        assert isinstance(global_params, dict)
        return global_params
    
        


