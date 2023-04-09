

from concurrent.futures import ThreadPoolExecutor
import grpc
import json
import traceback
import threading
import uuid
import sys
import grpc
from types import SimpleNamespace
from typing import Tuple, List, Union
from grpc import _common
import sys
import os
import asyncio
from copy import deepcopy
import commune
from .serializer import Serializer

class VirtualModule(commune.Module):
    def __init__(self, module: str ='ReactAgentModule', include_hiddden: bool = False):
        
        self.synced_attributes = []
        '''
        VirtualModule is a wrapper around a Commune module.
        
        Args:f
            module (str): Name of the module.
            include_hiddden (bool): If True, include hidden attributes.
        '''
        if isinstance(module, str):
            import commune
            self.module_client = commune.connect(module)
        else:
            self.module_client = module
        self.sync_module_attributes(include_hiddden=include_hiddden)
      
    def remote_call(self, remote_fn: str, *args, return_future= False, timeout=20, **kwargs):
        
    
        if return_future:
            return self.module_client.async_forward(fn=remote_fn, args=args, kwargs=kwargs, timeout=timeout)
        else:
            return self.module_client(fn=remote_fn, args=args, kwargs=kwargs, timeout=timeout)
            
    def sync_module_attributes(self, include_hiddden: bool = False):
        '''
        Syncs attributes of the module with the VirtualModule instance.
        
        Args:
            include_hiddden (bool): If True, include hidden attributes.
        '''
        from functools import partial
                
        for attr in self.module_client.server_functions:
            # continue if attribute is private and we don't want to include hidden attributes
            if attr.startswith('_') and (not include_hiddden):
                continue
            
            
            # set attribute as the remote_call
            setattr(self, attr,  partial(self.remote_call, attr))
            self.synced_attributes.append(attr)
            
            
    def __call__(self, *args, **kwargs):
        return self.remote_call(*args, **kwargs)
    def __getattr__(self, key):
        if key in ['synced_attributes', 'module_client', 'remote_call', 'sync_module_attributes'] :
            return getattr(self, key)
        
        return  self.module_client(fn='getattr', args=[key])
