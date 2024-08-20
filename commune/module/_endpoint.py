from typing import *
class Endpoint:

    def set_reference_module(self, module='storage', **kwargs):
        if not hasattr(self, 'reference_module'):
            reference_module = self.get_module(module)(**kwargs)
            my_endpoints = self.endpoints
            for k in reference_module.endpoints:
                if k in my_endpoints:
                    c.print(f'Endpoint {k} already exists in {self.class_name()}')
                self.add_endpoint(k, getattr(reference_module, k))
            self.reference_module = reference_module
        return {'success': True, 'msg': 'Set reference module', 'module': module, 'endpoints': self.endpoints}
        
    def add_endpoint(self, name, fn):
        if name in ['whitelist', 'blacklist']:
            print(f'Cannot add {name} as an endpoint')
            return {'success':False, 'message':f'Cannot add {fn} as an endpoint'}
        setattr(self, name, fn)
        self.endpoints.append(name)
        assert hasattr(self, name), f'{name} not added to {self.__class__.__name__}'
        return {'success':True, 'message':f'Added {fn} to {self.__class__.__name__}'}



    def is_endpoint(self, fn) -> bool:
        if isinstance(fn, str):
            fn = getattr(self, fn)
        return hasattr(fn, '__metadata__')


    
    def get_endpoints(self, search: str =None , helper_fn_attributes = ['helper_functions', 'whitelist']):
        endpoints = []
        for k in helper_fn_attributes:
            if hasattr(self, k):
                endpoints += getattr(self, k) 
        for f in dir(self):
            try:
                if not callable(getattr(self, f)) or  (search != None and search not in f):
                    continue
                fn_obj = getattr(self, f) # you need to watchout for properties
                is_endpoint = hasattr(fn_obj, '__metadata__')
                if is_endpoint:
                    endpoints.append(f)
            except Exception as e:
                print(f'Error in get_endpoints: {e} for {f}')
        return sorted(list(set(endpoints)))

    get_whitelist = whitelist_functions  = endpoints = get_endpoints

    

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
    


    def metadata(self, to_string=False):
        if hasattr(self, '_metadata'):
            return self._metadata
        metadata = {}
        metadata['schema'] = self.schema()
        metadata['description'] = self.description
        metadata['urls'] = {k: v for k,v in self.urls.items() if v != None}
        if to_string:
            return self.python2str(metadata)
        self._metadata =  metadata
        return metadata

    def info(self , 
             module = None,
             lite_features = ['name', 'address', 'schema', 'key', 'description'],
             lite = True,
             cost = False,
             **kwargs
             ) -> Dict[str, Any]:
        '''
        hey, whadup hey how is it going
        '''
        info = self.metadata()
        info['name'] = self.server_name or self.module_name()
        info['address'] = self.address
        info['key'] = self.key.ss58_address
        return info
    
    @classmethod
    def is_public(cls, fn):
        if not cls.is_endpoint(fn):
            return False
        return getattr(fn, '__metadata__')['public']


    urls = {'github': None,
             'website': None,
             'docs': None, 
             'twitter': None,
             'discord': None,
             'telegram': None,
             'linkedin': None,
             'email': None}