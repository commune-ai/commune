import commune as c
import pandas as pd

class Test:

    
    def __init__(self, module='chain', networkk='test', test_key='test'): 
        self.chain = c.module(module)(network=networkk)
        self.key = c.get_key(test_key)

    def test_global_params(self):
        global_params = self.chain.global_params()
        assert isinstance(global_params, dict)
        return {'msg': 'global_params test passed', 'success': True}

    def test_subnet(self, subnet=0):
        subnets = self.chain.subnets()
        assert isinstance(subnets, pd.DataFrame)
        return {'msg': 'subnet_params test passed', 'success': True, 'subsets': subnets}


    def test_module_params(self, keys=['dividends', 'incentive'], subnet=0):
        key = self.chain.keys(subnet)[0]
        module_info = self.chain.module(key, subnet=subnet)
        assert isinstance(module_info, dict)
        for k in keys:
            assert k in module_info, f'{k} not in {module_info}'

        return {'msg': 'module_params test passed', 'success': True, 'module_info': module_info}

    def test_transfer(self, from_key = 'test', to_key = 'test2', amount = 0.1, margin=1, safety=False, subnet=0):
        assert self.chain.network == 'test', f'Network {self.chain.network} is not test'
        to_key_address = c.get_key(to_key).key_address
        from_key_address = c.get_key(from_key).key_address
        balance_before = self.chain.balance(to_key_address)
        tx = self.chain.transfer(from_key, amount , to_key_address, safety=safety)
        balance_after = self.chain.balance(to_key_address)
        return {'msg': 'transfer test passed', 'success': True}

    def test_register_module(self, key='test', subnet=3):
        # generate random_key = 
        key = c.get_key(name)
        self.chain.register(key, subnet=subnet)
        rm_key = c.rm_key(name)
        return self.chain.module(key.ss58_address)
        

    def test_substrate(self):
        for i in range(3):
            t1 = c.time()
            c.print(self.chain.substrate)
            t2 = c.time()
            c.print(f'{t2-t1:.2f} seconds')
        return {'msg': 'substrate test passed', 'success': True}





