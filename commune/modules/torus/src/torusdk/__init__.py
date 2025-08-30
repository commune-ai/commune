"""
The Torus CLI library package.

Submodules:
    * `torus.client`: A lightweigh yet faster client for the Torus Network.
    * `.compat`: Compatibility layer for the *classic* `commune` library.
    * `.types`: Torus common types.
    * `.key`: Key related functions.

.. include:: ../../README.md
"""

import importlib.metadata

if not __package__:
    __version__ = "0.0.0"
else:
    __version__ = importlib.metadata.version(__package__)
