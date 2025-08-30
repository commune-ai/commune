import commune as c
import json
import os
from typing import List, Dict, Union, Optional, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

print = c.print

class SumMod:
    """
    Module to summarize all files in a folder using LLM-based semantic understanding.
    
    This module processes all files in a directory and provides summaries based on
    a query, with support for caching and parallel processing.
    """
    sum_folder = c.mod('sum.folder')()
    def forward(self, module='base', **kwargs):
        is_folder_module = c.is_folder_module(module)
        if  is_folder_module:
            return c.mod('sum.folder')().forward(path=c.dirpath(module), **kwargs)
        else:
            return c.mod('sum.file')().forward(path=c.filepath(module), **kwargs)
