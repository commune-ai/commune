
import os
from functools import partial
from typing import List, Dict

class Routes:

    @classmethod
    def routes_path(cls):
        return cls.dirpath() + '/routes.yaml'

    @classmethod
    def has_routes(cls):
        
        return os.path.exists(cls.routes_path()) or (hasattr(cls, 'routes') and isinstance(cls.routes, dict)) 
    
    @classmethod
    def routes(cls):
        if not cls.has_routes():
            return {}
        return cls.get_yaml(cls.routes_path())

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

        def fn_generator(*args, fn, module, **kwargs):
            module = cls.module(module)
            fn_type = module.classify_fn(fn)
            if fn_type == 'self':
                module = module()
            else:
                module = module
            return getattr(module, fn)(*args, **kwargs)

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
                fn_obj = partial(fn_generator, fn=from_fn, module=m )
                fn_obj.__name__ = to_fn
                # set the function to the current module
                setattr(cls, to_fn, fn_obj)
        t1 = cls.time()
        cls.print(f'enabled routes in {t1-t0} seconds', verbose=verbose)
        cls.routes_enabled = True
        return {'success': True, 'msg': 'enabled routes'}
    

    