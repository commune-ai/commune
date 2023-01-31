

from .module import Module
from .config import Config

for f in Module.get_class_methods():
    globals()[f] = getattr(Module, f)

