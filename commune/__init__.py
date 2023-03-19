

from .module import Module
from .config import Config as config


from rich.console import Console
__console__ = Console()
# Substrate ss58_format
__ss58_format__ = 42

# set the module functions as globals
for f in Module.get_class_methods():
    globals()[f] = getattr(Module, f)

