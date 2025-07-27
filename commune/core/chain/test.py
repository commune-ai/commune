import commune as c
import pandas as pd

class Test:

    
    def __init__(self, module='chain', networkk='test', test_key='test'): 
        self.chain = c.mod(module)(network=networkk)
        self.fund = self.chain.fund
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

    def test_transfer(self, from_key = 'test3', to_key = 'test0', amount = 1.0, margin=0.1, safety=False, max_period=10,subnet=0):
        assert self.chain.network == 'test', f'Network {self.chain.network} is not test'
        to_key_address = c.get_key(to_key).key_address
        from_key_address = c.get_key(from_key).key_address

        from_balance = self.chain.balance(from_key_address)
        to_balance = self.chain.balance(to_key_address)
        print(f'from_balance: {from_balance}, to_balance: {to_balance}, amount: {amount}, margin: {margin}')
        if from_balance < amount + margin and to_balance >= amount + margin:
            c.print(f'Insufficient balance for transfer: {from_balance} < {amount + margin}, but {to_balance} is available')
            from_key_address, to_key_address = to_key_address, from_key_address
            from_key = to_key
        t0 = c.time()
        while t0 + max_period > c.time():
            if self.chain.balance(from_key_address) >= amount + margin:
                break
            c.print(f'Waiting for balance to be sufficient: {self.chain.balance(from_key_address)} < {amount + margin}')
            c.sleep(1)
        tx = self.chain.transfer(from_key, amount , to_key_address, safety=safety)
        to_balance_after = self.chain.balance(to_key_address)
        print(f'to_balance: {to_balance_after}, from_balance: {from_balance}')

        

        assert abs(to_balance_after - (to_balance + amount)) < 0.1, f'Balance after ot match expected {from_balance + amount}'
        return {'msg': 'transfer test passed', 'success': True}



    def test_register_module(self, key='test3', subnet=3):
        # generate random_key = 
        self.chain.deregister(key, subnet=subnet)
        self.chain.register(key, subnet=subnet)
        return self.chain.module(key)

        

    def test_substrate(self):
        for i in range(3):
            t1 = c.time()
            c.print(self.chain.substrate)
            t2 = c.time()
            c.print(f'{t2-t1:.2f} seconds')
        return {'msg': 'substrate test passed', 'success': True}





