import commune as c 
from typing import *
import json
import os

class ProcessManager:
    desc = 'docker'
    def __init__(self, *args, **kwargs):
        self.docker = Docker(*args, **kwargs)
        for k in dir(self.docker):
            print(k)
            if not k.startswith('__'):
                setattr(self, k, getattr(self.docker, k))
