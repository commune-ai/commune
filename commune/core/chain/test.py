import commune as c


class Test:

    
    def __init__(self, module='chain', params={"network": 'test'}, test_key='test'): 
        self.chain = c.module(module)(**params)
        self.key = c.get_key(test_key)

    def test_global_params(self):
        global_params = self.chain.global_params()
        assert isinstance(global_params, dict)
        return {'msg': 'global_params test passed', 'success': True}

    def test_subnet_params(self, subnet=0):
        self = c.module('chain')()
        subnet_params = self.subnet_params(subnet=subnet)
        assert isinstance(subnet_params, dict), f'{subnet_params} is not a dict'
        return {'msg': 'subnet_params test passed', 'success': True}


    def test_module_params(self, keys=['dividends', 'incentive'], subnet=0):
        key = self.chain.keys(subnet)[0]
        module_info = self.chain.module(key, subnet=subnet)
        assert isinstance(module_info, dict)
        for k in keys:
            assert k in module_info, f'{k} not in {module_info}'

        return {'msg': 'module_params test passed', 'success': True, 'module_info': module_info}

    def test_substrate(self):
        for i in range(3):
            t1 = c.time()
            c.print(self.chain.substrate)
            t2 = c.time()
            c.print(f'{t2-t1:.2f} seconds')
        return {'msg': 'substrate test passed', 'success': True}





