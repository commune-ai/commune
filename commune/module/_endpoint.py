
class Endpoint:

    def set_reference_module(self, module='storage', **kwargs):
        if not hasattr(self, 'reference_module'):
            reference_module = self.get_module(module)(**kwargs)
            my_endpoints = self.endpoints()
            for k in reference_module.endpoints():
                if k in my_endpoints:
                    c.print(f'Endpoint {k} already exists in {self.class_name()}')
                self.add_endpoint(k, getattr(reference_module, k))
            self.reference_module = reference_module
        return {'success': True, 'msg': 'Set reference module', 'module': module, 'endpoints': self.endpoints()}
        
    def add_endpoint(self, name, fn):
        if name in ['whitelist', 'blacklist']:
            print(f'Cannot add {name} as an endpoint')
            return {'success':False, 'message':f'Cannot add {fn} as an endpoint'}
        setattr(self, name, fn)
        self.whitelist.append(name)
        assert hasattr(self, name), f'{name} not added to {self.__class__.__name__}'
        return {'success':True, 'message':f'Added {fn} to {self.__class__.__name__}'}



    def is_endpoint(self, fn) -> bool:
        if isinstance(fn, str):
            fn = getattr(self, fn)
        return hasattr(fn, '__metadata__')


    
    def endpoints(self, search=None, include_helper_functions = True):
        endpoints = []  
        if include_helper_functions:
            endpoints += self.helper_functions

        for f in dir(self):
            try:
                if not callable(getattr(self, f)):
                    continue

                if search != None:
                    if search not in f:
                        continue
                fn_obj = getattr(self, f) # you need to watchout for properties
                is_endpoint = hasattr(fn_obj, '__metadata__')
                if is_endpoint:
                    endpoints.append(f)
            except:
                print(f)
        if hasattr(self, 'whitelist'):
            endpoints += self.whitelist
            endpoints = list(set(endpoints))

        return endpoints

    get_whitelist = whitelist_functions = endpoints
    

    def cost_fn(self, fn:str, args:list, kwargs:dict):
        return 1
    


    @classmethod
    def endpoint(cls, 
                 cost=1, # cost per call 
                 user2rate : dict = None, 
                 rate_limit : int = 100, # calls per minute
                 timestale : int = 60,
                 public:bool = False,
                 cost_keys = ['cost', 'w', 'weight'],
                 **kwargs):
        
        for k in cost_keys:
            if k in kwargs:
                cost = kwargs[k]
                break

        def decorator_fn(fn):
            metadata = {
                **cls.fn_schema(fn),
                'cost': cost,
                'rate_limit': rate_limit,
                'user2rate': user2rate,   
                'timestale': timestale,
                'public': public,            
            }
            import commune as c
            fn.__dict__['__metadata__'] = metadata

            return fn

        return decorator_fn
    


