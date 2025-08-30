import commune as c
import json
import os
from typing import List, Dict, Union, Optional, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

print = c.print

class SumFolder:
    """
    Module to summarize all files in a folder using LLM-based semantic understanding.
    
    This module processes all files in a directory and provides summaries based on
    a query, with support for caching and parallel processing.
    """
    def __init__(self, **kwargs):
        self.sum_file = c.mod('sum.file')(**kwargs)

    def forward(self, path: str = './', **kwargs) -> List[str]:
        """
        Summarize the contents of a folder.
        """
        files = c.files(path)
        results = {}
        n = len(files)
        for i, file in enumerate(files):
            print(f"Summarizing file {i + 1}/{n}: {file}")
            results[file] = self.sum_file.forward(path=file, **kwargs)
        return results