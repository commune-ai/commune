

from .module import Module
from .config import Config
from .cli import cli
config = Config


from rich.console import Console
console = Console()

# Substrate ss58_format
__ss58_format__ = 42
# set the module functions as globals
for f in Module.get_class_methods():
    print(f)
    globals()[f] = getattr(Module, f)

# # we want the key and wallet abstraction to be global
# for k in ['key', 'wallet']:
#     globals()[k] = getattr(Module, 'get_' + k)