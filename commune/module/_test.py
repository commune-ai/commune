class Test:

    @classmethod
    def has_test_module(cls, module=None):
        module = module or cls.module_name()
        return cls.module_exists(cls.module_name() + '.test')
    
    @classmethod
    def test(cls,
              module=None,
              timeout=42, 
              trials=3, 
              parallel=False,
              ):
        module = module or cls.module_name()

        if cls.has_test_module(module):
            cls.print('FOUND TEST MODULE', color='yellow')
            module = module + '.test'
        self = cls.module(module)()
        test_fns = self.test_fns()
        print(f'testing {module} {test_fns}')

        def trial_wrapper(fn, trials=trials):
            def trial_fn(trials=trials):

                for i in range(trials):
                    try:
                        return fn()
                    except Exception as e:
                        print(f'Error: {e}, Retrying {i}/{trials}')
                        cls.sleep(1)
                return False
            return trial_fn
        fn2result = {}
        if parallel:
            future2fn = {}
            for fn in self.test_fns():
                cls.print(f'testing {fn}')
                f = cls.submit(trial_wrapper(getattr(self, fn)), timeout=timeout)
                future2fn[f] = fn
            for f in cls.as_completed(future2fn, timeout=timeout):
                fn = future2fn.pop(f)
                fn2result[fn] = f.result()
        else:
            for fn in self.test_fns():
                fn2result[fn] = trial_wrapper(getattr(self, fn))()

                
        return fn2result
