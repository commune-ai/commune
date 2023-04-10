
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")
# Reset warning filters to their default state (optional)
warnings.resetwarnings()

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
        args, kwargs = self.parse_args()        
        if len(args)> 0:
    
            if len(args[0].split(':')) == 2:
                # commune module:fn args kwargs 
                module_fn = args.pop(0)
                module, fn = module_fn.split(':')
                module = commune.connect(module)
                result = module.remote_call(fn, *args,**kwargs)
            else:
                fn = args.pop(0)
                result = getattr(commune, fn)(*args, **kwargs)
            self.print(result) 
        
        else:
            self.print("No command given", color='red')
            
            
    
            
            
