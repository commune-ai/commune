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
        key = None,
    ) -> 'Server':
        if isinstance(module, str):
            module = c.module(module)()
        elif module is None:
            module = c.module('module')()
        self.module = module
        if name == None:
            name = self.module.name()

        c.print('MODULE',self.module, color='green')
        self.name = name
        self.timeout = timeout
        self.verbose = verbose
        self.serializer = c.serializer()
        self.ip = c.resolve_ip(ip, external=False)  # default to '0.0.0.0'
        self.port = c.resolve_port(port)
        self.address = f"{self.ip}:{self.port}"
        self.set_api()
        self.key = c.get_key(name)


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
        import uvicorn

        self.app = FastAPI()


        @self.app.post("/{fn}")
        async def forward_wrapper(fn, input:dict[str, str]):
            for k,v in input.items():
                input[k] = self.serializer.deserialize(v)
            args = input.get('args', [])
            kwargs = input.get('kwargs', {})

            result =  self.forward(fn=fn, args=args, kwargs=kwargs)

            return {'data':self.serializer.serialize(result)}
        
        c.register_server(self.name, self.ip, self.port)
        uvicorn.run(self.app, host=self.ip, port=self.port)

    def forward(self, fn: str, args: List = None, kwargs: Dict = None, **extra_kwargs):
        try: 
            if args is None:
                args = []
            if kwargs is None:
                kwargs = {}
            obj = getattr(self.module, fn)
            if callable(obj):
                response = obj(*args, **kwargs)
            else:
                response = obj
            return response
        except Exception as e:
            response = {'error': str(e)}
            c.print(e, color='red')
        return response


    def __del__(self):
        c.deregister_server(self.name)