import commune as c

def test_global_params():
    self = c.module('subspace')()
    global_params = self.global_params(fmt='dict')
    assert isinstance(global_params, dict)
    return {'msg': 'global_params test passed', 'success': True}

def test_subnet_params(netuid=0):
    self = c.module('subspace')()
    subnet_params = self.subnet_params(netuid=0)
    assert isinstance(subnet_params, dict)
    subnet_names = self.subnet_names()
    assert isinstance(subnet_names, list) and len(subnet_names) > 0
    subnet2netuid = self.subnet2netuid()
    assert isinstance(subnet2netuid, dict) and len(subnet2netuid) > 0
    namespace = self.namespace(netuid=netuid)
    assert isinstance(namespace, dict)
    return {'msg': 'subnet_params test passed', 'success': True}


def test_module_params(keys=['dividends', 'incentive'], netuid=0):
    self = c.module('subspace')()
    key = self.keys(netuid=netuid)[0]
    module_info = self.get_module(key, netuid=netuid)
    assert isinstance(module_info, dict)
    for k in keys:
        assert k in module_info, f'{k} not in {module_info}'

    return {'msg': 'module_params test passed', 'success': True, 'module_info': module_info}


def test_substrate():
    self = c.module('subspace')()
    for i in range(3):
        t1 = c.time()
        c.print(self.substrate)
        t2 = c.time()
        c.print(f'{t2-t1:.2f} seconds')
    return {'msg': 'substrate test passed', 'success': True}


    



