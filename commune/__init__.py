

from .module import Module
from .config import Config as config
# set the module functions as globals
for f in Module.get_class_methods():
    globals()[f] = getattr(Module, f)

