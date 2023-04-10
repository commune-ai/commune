
import argparse
import commune
from typing import List, Optional
import json

# Turn off rich console locals trace.
from rich.traceback import install
install(show_locals=False)

class CLI(commune.Module):
    """
    Create and init the CLI class, which handles the coldkey, hotkey and tao transfer 
    """
    def __init__(
            self,
            config: commune.Config = None,

        ) :
        args, kwargs = self.parse_args()
        self.print(f'{args} {kwargs}')
        
        if len(args)> 0:
    
            if len(args[0].split(':')) == 2:
                # commune module:fn args kwargs 
                module_fn = args.pop(0)
                module, fn = module_fn.split(':')
                module = commune.connect(module)
                result = module.remote_call(fn, *args,**kwargs)
            else:
                fn = args.pop(0)
                self.print(args, fn)
                result = getattr(commune, fn)(*args, **kwargs)
            self.print(result) 
        
        else:
            self.print("No command given", color='red')
            
            
    
            
            
