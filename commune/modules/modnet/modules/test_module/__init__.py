"""
Test Module for Mod-Net Module Registry

This package provides a test module that demonstrates integration with the
Module Registry system and IPFS metadata storage.
"""

from .module import TestModule, main

__version__ = "1.0.0"
__author__ = "mod-net-developer@example.com"
__all__ = ["TestModule", "main"]

# Commune module exports
module = TestModule
