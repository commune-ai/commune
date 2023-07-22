import argparse
import asyncio
import os
import signal
import sys
import time
from concurrent import futures
from typing import Dict, List, Callable, Optional, Tuple, Union

import commune as c
import torch
from loguru import logger
from munch import Munch

from http.server import BaseHTTPRequestHandler, HTTPServer





class HTTPServer(c.Module):
    def __init__(
        self,
        module: Union[c.Module, object] = None,
        name: str = None,
        ip: Optional[str] = None,
        port: Optional[int] = None,
        timeout: Optional[int] = 4,
        verbose: bool = True,
        whitelist: List[str] = None,
        blacklist: List[str] = None,
    ) -> 'Server':
        self.module = c.resolve_module(module)()
        if name == None:
            name = self.module.name()
        self.name = name
        self.timeout = timeout
        self.verbose = verbose

        self.ip = c.resolve_ip(ip, external=False)  # default to '0.0.0.0'
        self.port = c.resolve_port(port)
        self.address = f"{self.ip}:{self.port}"
        self.set_api()


    def state_dict(self) -> Dict:
        return {
            'ip': self.ip,
            'port': self.port,
            'address': self.address,
            'timeout': self.timeout,
            'verbose': self.verbose,
        }


    def test(self):
        r"""Test the HTTP server.
        """
        # Test the server here if needed
        c.print(self.state_dict(), color='green')
        return self

    def set_api(self):

        from fastapi import FastAPI
        import requests

        self.app = FastAPI()

        @self.app.get("/{fn}")
        def forward_wrapper(fn: str = None, args: List = None, kwargs: Dict = None):
            return self.forward(fn, args, kwargs)
        
        import uvicorn
        c.register_server(self.name, self.ip, self.port)
        uvicorn.run(self.app, host=self.ip, port=self.port)




    def forward(self, fn: str, args: List = None, kwargs: Dict = None, **extra_kwargs):
        try: 
            if args is None:
                args = []
            if kwargs is None:
                kwargs = {}
            c.print(f'FORWARDING {fn} {args} {kwargs}, {self.module}', color='green')
            response = getattr(self.module, fn)(*args, **kwargs)
        except Exception as e:
            response = {'error': str(e)}

        return response
    


    def __del__(self):
        c.deregister_server(self.name)