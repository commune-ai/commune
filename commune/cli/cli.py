

import argparse
import commune
from typing import List, Optional
import json


class CLI(commune.Module):
    """
    Create and init the CLI class, which handles the coldkey, hotkey and tao transfer 
    """
    def __init__(
            self,
            config: commune.Config = None,

        ) :
        commune.new_event_loop(True)
        self.module = commune.Module()
        args, kwargs = self.parse_args()

        

        module_list = commune.module_list()

        fn = None
        module = None
        if len(args)> 0:
    
            namespace = self.namespace(update=False)
            
 

            module_list = self.module_list()
            functions = self.functions(args[0], include_module=True)
            module_options = (list(namespace.values()) + list(namespace.keys()))
            candidates = dict(
                            functions=[f for f in functions if f == args[0]],
                            modules=[m for m in module_list if m == args[0]],
                            servers=[s for s in module_options if args[0] == s],
            )
            
            if len(args[0].split('.')) > 1:
                new_servers = [f for f in module_options if '.'.join(args[0].split('.')[:-1]) == f]
                if len(new_servers)>0:
                    candidates['servers'] = new_servers
                    module = new_servers[0]
                    fn = args[0].split('.')[-1]
                    
                new_modules = [f for f in module_list if '.'.join(args[0].split('.')[:-1]) == f]
                if len(new_modules)>0:
                    candidates['modules'] = new_modules
                    module = new_modules[0]
                    fn = args[0].split('.')[-1]
                    
                    
            # fn_obj = getattr(module, fn)
            if len(candidates['modules'])>0:
                if module == None:
                    module = args.pop(0)
                else:
                    args.pop(0)
                fn = fn if fn != None else args.pop(0)
                module = commune.module(module)
            elif len(candidates['servers'])>0:
                if module == None:
                    module = args.pop(0)
                else:
                    args.pop(0)
                fn = fn if fn != None else args.pop(0)
                module = commune.connect(module)

            elif len(candidates['functions'])>0:
                module = self.module
                fn = fn if fn != None else  args.pop(0)
                fn = candidates['functions'][0]
            else: 
                raise Exception(f'No module, function or server found for {args[0]}')
        
        result = getattr(module, fn)
        if callable(result):
            result = result(*args, **kwargs)


        self.print(result)
    
    def catch_ip(self):
        result = None
        if len(args[0].split(':')) == 2:
            # commune module:fn args kwargs 
            module_fn = args.pop(0)
            module, fn = module_fn.split(':')
            module = commune.connect(module)
            result = module.remote_call(fn, *args,**kwargs)
            
            
        return result
