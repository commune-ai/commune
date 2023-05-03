

from .module import Module
# call it whatever you want, but it's the same thing
Block = Lego = Module
from .config import Config
from .cli import cli
from .model import Model
config = Config



# Substrate ss58_format
__ss58_format__ = 42
# set the module functions as globals
for k,v in Module.__dict__.items():
    globals()[k] = v

for f in Module.get_class_methods():
    globals()[f] = getattr(Module, f)
    
key = Module.key
address = Module.address
user = Module.user
def signin(usr: str, pwd: str):
    global key, address, user
    user = usr 
    key = Module.signin(usr, pwd)
    address = key.address

