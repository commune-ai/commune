       
import commune as c
class Test(c.Module):
    def test(self,
              modules=['server', 
                       'key', 
                       'namespace', 
                       'executor', 
                       'vali'],
              timeout=40):
        futures = []
        results = []
        if cls.module_path() == 'module':
            for module_name in modules:
                module = c.module(module_name)
                assert hasattr(module, 'test'), f'Module {module_name} does not have a test function'
                futures.append(c.submit(module.test))
            results = c.wait(futures, timeout=timeout)
            results = dict(zip(modules, results))
            for module_name, result in results.items():
                if c.is_success(result):
                    results[module_name] = 'success'
                else  :
                    results[module_name] = 'failure'
        else:
            module_fns = c.fns()
            fns = [getattr(self,f) for f in self.fns() if f.startswith('test_') and not (f in module_fns and self.module_path() != 'module')]
            c.print(f'Running {len(fns)} tests')
            for fn in fns:
                results += [c.submit(fn)]
            
            results = c.wait(results, timeout=timeout)

        return results
       