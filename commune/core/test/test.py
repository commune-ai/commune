    
import commune as c
class Test(c.Module):
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
        try:
            if self.module_exists(module + '.test'):
                return True
            else: 
                fns = c.fns(module)
                if 'test' in fns or any([fn.startswith('test_') for fn in fns]):
                    return True
        except Exception as e:
            if verbose:
                c.print(f'Error checking test {module}: {e}')
        return False
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

        obj = self.module(module)()
        test_fns = []
        for fn in dir(obj):
            if fn.startswith('test_') or fn == 'test':
                fn_obj = getattr(obj, fn)


                test_fns.append(fn_obj)
        return test_fns


    def test_mods(self, search=None, verbose=False, **kwargs):
        test_mods = []
        mods =  c.mods(search=search, **kwargs)
        for m in mods:
            if self.has_test_module(m, verbose=verbose):
                test_mods.append(m)
        return test_mods