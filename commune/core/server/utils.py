
import os

def shortkey(x: str, n=6) -> str:
    return x[:n] + '..' 

def abspath(path: str) -> str:
    """Get the absolute path of a file or directory"""
    return os.path.abspath(os.path.expanduser(path))