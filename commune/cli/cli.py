
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
            args: Optional[List[str]] = None, 
            config: commune.Config = None,

        ) :

        # config = self.set_config(config)
        # self.argparse()
        getattr(self, args[0])(*args[1:])
        
    def modules(self, *args, **kwargs):
        commune.print(commune.modules())
    def functions(self, *args, **kwargs):
        commune.print(commune.functions())
        
    def schema(self, *args, **kwargs):
        if len(args) == 0:
            schema = commune.print(commune.get_function_schema_map())
        elif len(args) == 1:
            schema = commune.print(commune.get_function_schema_map(args[0]))

        # commune.print(schema)
    def tree(self, *args, **kwargs):
        commune.print(commune.module_list())

    def list(self, *args, **kwargs):
        commune.print(commune.module_list())

    def launch(self, *args, **kwargs):
        commune.print(commune.launch(*args))

        
