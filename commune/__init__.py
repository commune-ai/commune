
from .module import Module
from .cli import cli
from functools import  partial
from .vali.vali import Vali

# set the module functions as globals
for k,v in Module.__dict__.items():
    globals()[k] = v

# for f in :
#     globals()[f] = getattr(Module, f)
    
for f in Module.class_functions() + Module.static_functions():
    globals()[f] = getattr(Module, f)

for f in Module.self_functions():
    def wrapper_fn(f, *args, **kwargs):
        fn = getattr(Module(), f)
        return fn(*args, **kwargs)
    globals()[f] = partial(wrapper_fn, f)

globals()['cli'] = cli

c = Block = Lego = M = Module  # alias c.Module as c.Block, c.Lego, c.M

