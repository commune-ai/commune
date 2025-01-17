from .module import Module  # the module module

M = c = Block = Agent = Module  # alias c.Module as c.Block, c.Lego, c.M
from .vali.vali import Vali  # the vali module
from .server.server import Server  # the server module
from .server.client import Client  # the client module
from .key.key import Key  # the key module

# set the module functions as globalsw
c.add_to_globals(globals())
network = c.network

v = Vali 
s = Server
k = Key
# set the module functions as globals
key = (
  c.get_key
) # override key function with file key in commune/key.py TODO: remove this line with a better solution

