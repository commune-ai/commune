

from .module import Module
from .config import Config
from .cli import cli
config = Config


# Substrate ss58_format
__ss58_format__ = 42
# set the module functions as globals
for f in Module.get_class_methods():
    globals()[f] = getattr(Module, f)
