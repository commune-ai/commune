
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
                    
        if len(args)> 0:
            fn = args.pop(0)
            self.print(getattr(commune, fn)(*args, **kwargs))
        
        else:
            self.print("No command given", color='red')
            