
from .module import Module
Block = Lego = Module

# call it whatever you want, but it's the same thing
from .module.config import Config
config = Config
from .cli import cli
# from .modules.subspace import subspace
# from .model import Model

# set the module functions as globals
for k,v in Module.__dict__.items():
    globals()[k] = v

for f in Module.get_class_methods() + Module.get_static_methods():
    globals()[f] = getattr(Module, f)
    