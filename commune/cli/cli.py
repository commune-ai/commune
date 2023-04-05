
import argparse
import commune
from typing import List, Optional
# Turn off rich console locals trace.
from rich.traceback import install
install(show_locals=False)

class CLI(commune.Module):
    """
    Create and init the CLI class, which handles the coldkey, hotkey and tao transfer 
    """
    def __init__(
            self,
            args: Optional[List[str]] = None, 
            config: commune.Config = None,

        ) :

        config = self.set_config(config)
