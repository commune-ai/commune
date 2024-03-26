
from .module import Module
Block = Lego = M = Module  # alias c.Module as c.Block, c.Lego, c.M
from .cli import cli
# from .modules.subspace import subspace
# from .model import Model

# set the module functions as globals
for k,v in Module.__dict__.items():
    globals()[k] = v

for f in Module.get_class_methods() + Module.get_static_methods():
    globals()[f] = getattr(Module, f)
    
for f in Module.get_self_methods():
    globals()[f] = lambda *args, **kwargs: getattr(Module(), f)(*args, **kwargs)
    