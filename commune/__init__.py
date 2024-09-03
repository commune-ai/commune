


from .module import Module
from functools import  partial
# set the module functions as globals

# for f in :
#     globals()[f] = getattr(Module, f)
    

c = Block = Lego = M = Module  # alias c.Module as c.Block, c.Lego, c.M
c.add_to_globals(globals())