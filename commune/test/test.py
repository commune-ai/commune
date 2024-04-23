import commune as c

class Test(c.Module):


    modules = ['key', 'namespace', 'server', 'subspace', 'module']

    def test(self, timeout=60):
        modules = self.modules
        future2module = {}
        for module in modules:
            f = c.submit(c.module(module).test, timeout=timeout)
            future2module[f] = module
        
        results = {}
        for f in c.as_completed(future2module.keys(), timeout=timeout):
            module = future2module[f]
            results[module] = f.result()
            c.log('module %s finished' % module)

        return results
            
