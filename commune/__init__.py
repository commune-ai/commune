

from .module import Module
from .config import Config as config


from rich.console import Console
console = Console()

# Substrate ss58_format
__ss58_format__ = 42
# set the module functions as globals
for f in Module.get_class_methods():
    globals()[f] = getattr(Module, f)

# we want the key abstraction to be global
globals()['key'] =  Module.get_key

