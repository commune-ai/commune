    
import commune as c
class Test:
    description = """
    i test stuff
    """
    def forward(self, module=None, timeout=50, modules=[ 'server', 'vali','key', 'chain']):
        """
        Test the module 
        """
        if module == None:
            test_results ={}
            for m in modules:
                print(f'Testing module: {m}')
                test_results[m] = self.test(module=m, timeout=timeout)
            return test_results
        else:
            fn2result = {}
            fns = self.test_fns(module)
            for fn in fns:
                fn_name = fn.__name__
                try:
                    fn2result[fn_name] = fn()
                    print(f'TestResult({fn_name}): {fn2result[fn_name]}')
                except Exception as e:
                    c.print(f'TestError({e})')
                    fn2result[fn_name] = self.detailed_error(e)
            return fn2result


    def has_test_module(self, module, verbose=False):
        """
        Check if the module has a test module
        """
        return c.module_exists(module + '.test')

    def has_test_fns(self, module):
        return bool('test' in c.fns(module))
    def test_module(self, module='module', timeout=50):
        """
        Test the module
        """
        for fn in self.test_fns(test):
            print(f'Testing({fn})')
            future = self.submit(getattr(test, fn), timeout=timeout)
            futures += [future]
        results = []
        for future in self.as_completed(futures, timeout=timeout):
            print(future.result())
            results += [future.result()]
        return results

    testmod = test_module

    def test_fns(self, module='module'):
        if self.has_test_module(module):
            module = module + '.test'

        obj = c.mod(module)()
        test_fns = []
        for fn in dir(obj):
            if fn.startswith('test_') or fn == 'test':
                fn_obj = getattr(obj, fn)


                test_fns.append(fn_obj)
        return test_fns


    def has_test(self, module=None, verbose=False):
        """
        Check if the module has a test module or test functions
        """
        try:
            return self.has_test_module(module) or self.has_test_fns(module)
        except Exception as e:
            if verbose:
                c.print(f'Error checking tests for {module}: {e}')
        return False


    def test_mods(self, search=None, verbose=False, **kwargs):
        test_mods = []
        mods =  c.mods(search=search, **kwargs)
        for m in mods:
            if verbose:
                c.print(f'Checking module: {m}')
            if self.has_test(m, verbose=verbose):
                test_mods.append(m)
        return test_mods