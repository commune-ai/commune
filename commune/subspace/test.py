import commune as c

class Test(c.m('subspace')):

    def test_global_params(self):
        global_params = c.global_params(fmt='dict')
        assert isinstance(global_params, dict)


    def test_subnet_params(self):
        subnet_params_a = c.subnet_params(netuid=0)
        assert isinstance(subnet_params_a, dict)
        subnet_params_b = c.subnet_params(netuid='commune')
        assert isinstance(subnet_params_b, dict)
        assert c.hash(subnet_params_a) == c.hash(subnet_params_b), f'{subnet_params_a} != {subnet_params_b}'


        subnet_names = c.subnet_names()
        assert isinstance(subnet_names, list) and len(subnet_names) > 0
        subnet2netuid = c.subnet2netuid()
        assert isinstance(subnet2netuid, dict) and len(subnet2netuid) > 0
        

        namespace = self.namespace(netuid=subnet2netuid['commune'])
        assert isinstance(namespace, dict)
        namespace = self.namespace(netuid=0)
        assert isinstance(namespace, dict)
        namespace = self.namespace(netuid='all')
        assert isinstance(namespace, dict)
        namespace = self.namespace(network='subspace.commune')

        return {'msg': 'subnet_params test passed', 'success': True}
    


    def test_module_params(self, keys=['dividends', 'incentive']):
        self = c.module('subspace')()
        key = self.keys()[0]
        module_info = self.get_module(key)
        assert isinstance(module_info, dict)
        for k in keys:
            assert k in module_info, f'{k} not in {module_info}'

        return {'msg': 'module_params test passed', 'success': True, 'module_info': module_info}






        


