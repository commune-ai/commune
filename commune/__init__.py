

from .module import Module
# call it whatever you want, but it's the same thing
Block = Lego = Module
from .module.config import Config
from .modules.cli import cli
# from .modules.subspace import subspace
# from .model import Model
config = Config
import warnings
warnings.filterwarnings("ignore")


# set the module functions as globals
for k,v in Module.__dict__.items():
    globals()[k] = v

for f in Module.get_class_methods() + Module.get_static_methods():
    globals()[f] = getattr(Module, f)
    