

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
        module = commune.Module()
        args, kwargs = self.parse_args() 
        if len(args)> 0:
    

            # fn_obj = getattr(module, fn)
            if args[0] in commune.module_list():
                module = commune.get_module(args.pop(0))
            fn = args.pop(0)
            result = getattr(module, fn)(*args, **kwargs)
            # else:
            #     result = fn_obj
            self.print(result) 
        
        else:
            self.print("No command given", color='red')
            
    
    def catch_ip(self):
        result = None
        if len(args[0].split(':')) == 2:
            # commune module:fn args kwargs 
            module_fn = args.pop(0)
            module, fn = module_fn.split(':')
            module = commune.connect(module)
            result = module.remote_call(fn, *args,**kwargs)
            
            
        return result
