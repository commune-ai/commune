import commune as c


class Test:

    def test_global_params(self):
        self = c.module('subspace')()
        global_params = self.global_params()
        assert isinstance(global_params, dict)
        return {'msg': 'global_params test passed', 'success': True}

    def test_subnet_params(self, subnet=0):
        self = c.module('subspace')()
        subnet_params = self.subnet_params(subnet=subnet)
        assert isinstance(subnet_params, dict), f'{subnet_params} is not a dict'
        return {'msg': 'subnet_params test passed', 'success': True}


    def test_module_params(self, keys=['dividends', 'incentive'], subnet=0):
        self = c.module('subspace')()
        key = self.keys(subnet)[0]
        module_info = self.get_module(key, subnet=subnet)
        assert isinstance(module_info, dict)
        for k in keys:
            assert k in module_info, f'{k} not in {module_info}'

        return {'msg': 'module_params test passed', 'success': True, 'module_info': module_info}


    def test_substrate(self):
        self = c.module('subspace')()
        for i in range(3):
            t1 = c.time()
            c.print(self.substrate)
            t2 = c.time()
            c.print(f'{t2-t1:.2f} seconds')
        return {'msg': 'substrate test passed', 'success': True}


    def test(self):
        fns = [f for f in dir(self) if 'test_' in f]
        fn2result = {}
        fn2error = {}
        for fn in fns:
            try:
                result = getattr(self, fn)()
                fn2result[fn] = result
            except Exception as e:
                fn2error[fn] = {'success': False, 'msg': str(e)}
        if len(fn2error) > 0:
            raise Exception(f'Errors: {fn2error}')
           
        return fn2result





