


from .module import Module
# set the module functions as globals
c = Block = Lego = M = Module  # alias c.Module as c.Block, c.Lego, c.M
c.add_to_globals(globals())
# override key function with file key in commune/key.py
key = c.get_key
