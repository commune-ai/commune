from .mod import Mod
__all__ = ["__version__"]
__version__ = "0.0.1.beta0"
Module = Mod
Mod().add_globals(globals())
