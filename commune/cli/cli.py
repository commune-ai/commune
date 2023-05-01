

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

        if len(args)> 0:
    
            namespace = self.namespace(update=False)
            
 

            candidates = dict(
            functions=self.functions(args[0], include_module=True),
            modules=self.module_list(args[0]),
            servers=[s for s in (list(namespace.values()) + list(namespace.keys())) if args[0] in s],
            )
            # fn_obj = getattr(module, fn)
            if len(candidates['modules'])>0:
                module = args.pop(0)
                fn = args.pop(0)
                module = commune.get_module(module)
                result = getattr(module, fn)
                if callable(result):
                    result = result(*args, **kwargs)
            elif len(candidates['servers'])>0:
                module = commune.connect(module)
                fn = args.pop(0)
                result = getattr(module, fn)
                if callable(result):
                    result = result(*args, **kwargs)
            elif len(candidates['functions'])>0:
                fn = args.pop(0)
                fn = candidates['functions'][0]
                # module_info = module.info()
                
                result = getattr(self.module,fn)(*args, **kwargs)
            else: 
                raise Exception(f'No module, function or server found for {args[0]}')
            


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
