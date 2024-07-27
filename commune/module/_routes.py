
import os
from functools import partial
from typing import List, Dict


"""

Routes are a way to connect different modules together through the routes.yaml file.
The routes.yaml file is a dictionary that maps the module name to the function name.
There are 3 ways to define a route:

please look at self.resolve_to_from_fn_routes

way 1

```yaml
module_name:
    - function_name
```

For renaming the function name in the current module

way 2a 

```yaml
module_name:
    - [function_name, new_name]
```
or

way 2b
module_name:
    - {fn: 'function_name', name: 'new_name'}

"""

class Routes:

    @classmethod
    def routes_path(cls):
        return cls.dirpath() + '/routes.yaml'

    @classmethod
    def has_routes(cls):
        return os.path.exists(cls.routes_path()) or (hasattr(cls, 'routes') and isinstance(cls.routes, dict)) 
    
    route_cache = None
    @classmethod
    def routes(cls, cache=True):
        if cls.route_cache is not None and cache:
            return cls.route_cache 
        if not cls.has_routes():
            return {}
        
        routes =  cls.get_yaml(cls.routes_path())
        cls.route_cache = routes
        return routes

    #### THE FINAL TOUCH , ROUTE ALL OF THE MODULES TO THE CURRENT MODULE BASED ON THE routes CONFIG


    @classmethod
    def route_fns(cls):
        routes = cls.routes()
        route_fns = []
        for module, fns in routes.items():
            for fn in fns:
                if isinstance(fn, dict):
                    fn = fn['to']
                elif isinstance(fn, list):
                    fn = fn[1]
                elif isinstance(fn, str):
                    fn
                else:
                    raise ValueError(f'Invalid route {fn}')
                route_fns.append(fn)
        return route_fns
            

    @staticmethod
    def resolve_to_from_fn_routes(fn):
        '''
        resolve the from and to function names from the routes
        option 1: 
        {fn: 'fn_name', name: 'name_in_current_module'}
        option 2:
        {from: 'fn_name', to: 'name_in_current_module'}
        '''
        
        if type(fn) in [list, set, tuple] and len(fn) == 2:
            # option 1: ['fn_name', 'name_in_current_module']
            from_fn = fn[0]
            to_fn = fn[1]
        elif isinstance(fn, dict) and all([k in fn for k in ['fn', 'name']]):
            if 'fn' in fn and 'name' in fn:
                to_fn = fn['name']
                from_fn = fn['fn']
            elif 'from' in fn and 'to' in fn:
                from_fn = fn['from']
                to_fn = fn['to']
        else:
            from_fn = fn
            to_fn = fn
        
        return from_fn, to_fn
    

    @classmethod
    def enable_routes(cls, routes:dict=None, verbose=False):
        """
        This ties other modules into the current module.
        The way it works is that it takes the module name and the function name and creates a partial function that is bound to the module.
        This allows you to call the function as if it were a method of the current module.
        for example
        """
        my_path = cls.class_name()
        if not hasattr(cls, 'routes_enabled'): 
            cls.routes_enabled = False

        t0 = cls.time()

        # WARNING : THE PLACE HOLDERS MUST NOT INTERFERE WITH THE KWARGS OTHERWISE IT WILL CAUSE A BUG IF THE KWARGS ARE THE SAME AS THE PLACEHOLDERS
        # THE PLACEHOLDERS ARE NAMED AS module_ph and fn_ph AND WILL UNLIKELY INTERFERE WITH THE KWARGS
        def fn_generator( *args, module_ph, fn_ph, **kwargs):
            module_ph = cls.module(module_ph)
            fn_type = module_ph.classify_fn(fn_ph)
            module_ph = module_ph() if fn_type == 'self' else module_ph
            return getattr(module_ph, fn_ph)(*args, **kwargs)

        if routes == None:
            if not hasattr(cls, 'routes'):
                return {'success': False, 'msg': 'routes not found'}
            routes = cls.routes() if callable(cls.routes) else cls.routes
        for m, fns in routes.items():
            for fn in fns: 
                cls.print(f'Enabling route {m}.{fn} -> {my_path}:{fn}', verbose=verbose)
                # resolve the from and to function names
                from_fn, to_fn = cls.resolve_to_from_fn_routes(fn)
                # create a partial function that is bound to the module
                fn_obj = partial(fn_generator, fn_ph=from_fn, module_ph=m )
                # make sure the funciton is as close to the original function as possible
                fn_obj.__name__ = to_fn
                # set the function to the current module
                setattr(cls, to_fn, fn_obj)
        t1 = cls.time()
        cls.print(f'enabled routes in {t1-t0} seconds', verbose=verbose)
        cls.routes_enabled = True
        return {'success': True, 'msg': 'enabled routes'}
    

    def fn2module(cls):
        '''
        get the module of a function
        '''
        routes = cls.routes()
        fn2module = {}
        for module, fn_routes in routes.items():
            for fn_route in fn_routes:
                if isinstance(fn_route, dict):
                    fn_route = fn_route['to']
                elif isinstance(fn_route, list):
                    fn_route = fn_route[1]
                fn2module[fn_route] = module

            
        return fn2module
    

    def is_route(cls, fn):
        '''
        check if a function is a route
        '''
        return fn in cls.fn2module()
    