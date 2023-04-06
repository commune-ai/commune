
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
        
        self.print(f'{args} {kwargs}', color='green')
            
        if len(args)== 0:
            self.help()
        else:
            fn = args.pop(0)
            self.print(f"fn: {fn} args: {args}", color='green')
            self.print(getattr(commune, fn)(*args, **kwargs))
        
