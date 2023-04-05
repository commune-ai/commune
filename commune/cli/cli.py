
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
        if len(args)== 0:
            self.help()
        else:
            fn = args.pop(0)
            getattr(self, fn)(*args)
        
    def help(self):
        self.print(self.__doc__)
    def modules(self, *args, **kwargs):
        commune.print(commune.modules())
    def functions(self, *args, **kwargs):
        commune.print(commune.functions())
        
    def schema(self, *args, **kwargs):
        if len(args) == 0:
            schema = commune.schema()
        elif len(args) == 1:
            schema = commune.schema(args[0])

        commune.print(schema)
    def tree(self, *args, **kwargs):
        commune.print(commune.module_list())

    def list(self, *args, **kwargs):
        commune.print(commune.module_list())
        
    def servers(self, *args, **kwargs):
        commune.print(commune.servers())

    def pm2_list(self, *args, **kwargs):
        commune.print(commune.pm2_list())

    def launch(self, *args, **kwargs):
        return commune.launch(*args, **kwargs)

    def deploy_fleet(self, *args, **kwargs):
        return commune.deploy_fleet(*args)

        
