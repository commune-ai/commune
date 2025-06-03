    
import commune as c
class Test(c.Module):
    def forward(self, module=None, timeout=50, modules=[ 'server', 'vali','key', 'chain']):
        """
        Test the module 
        """
        if module == None:
            test_results ={}
            for m in modules:
                test_results[m] = self.test(m, timeout=timeout)
            return test_results
        module_obj = self.module(module)()
        if not hasattr(module, 'test') and self.module_exists(module + '.test'):
            module = module + '.test'
            module_obj = self.module(module)()
        fn2result = {}
        for i, fn in enumerate(self.test_fns(module_obj)):
            c.print(f'---Testing({fn})----')
            try:
                fn2result[fn] = getattr(module_obj, fn)()
            except Exception as e:
                c.print(f'TestError({e})')
                fn2result[fn] = self.detailed_error(e)
        return fn2result

    def test_module(self, module='module', timeout=50):
        """
        Test the module
        """

        if self.module_exists(module + '.test'):
            module = module + '.test'

        if module == 'module':
            module = 'test'
        Test = self.module(module)
        test_fns = [f for f in dir(Test) if f.startswith('test_') or f == 'test']
        test = Test()
        futures = []
        for fn in test_fns:
            print(f'Testing({fn})')
            future = self.submit(getattr(test, fn), timeout=timeout)
            futures += [future]
        results = []
        for future in self.as_completed(futures, timeout=timeout):
            print(future.result())
            results += [future.result()]
        return results

    testmod = test_module

    def test_fns(self, module=None):
        return [f for f in dir(self.module(module)) if f.startswith('test_') or f == 'test']
