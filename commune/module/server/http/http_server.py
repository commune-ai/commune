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


class HTTPRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        # Process the received data
        # ...

        self.wfile.write(b'OK')



class Server(c.Module):
    def __init__(
        self,
        module: Union[c.Module, object] = None,
        name: str = None,
        ip: Optional[str] = None,
        port: Optional[int] = None,
        timeout: Optional[int] = None,
        verbose: bool = True,
        whitelist: List[str] = None,
        blacklist: List[str] = None,
    ) -> 'Server':
        self.module = module
        self.name = c.resolve_name(name)
        self.timeout = timeout
        self.verbose = verbose
        self.ip = c.resolve_ip(ip, external=False)  # default to '0.0.0.0'
        self.port = c.resolve_port(port)
        self.address = f"{self.ip}:{self.port}"

        # Whether or not the server is running
        self.started = False
        self.set_stats()

        self.set_module_info()

    def set_module_info(self):
        self.module_info = self.module.info()
        self.allowed_functions = self.module_info['functions']
        self.allowed_attributes = self.module_info['attributes']

    def set_stats(self):
        self.stats = dict(
            call_count=0,
            total_bytes=0,
            time={},
        )

    def __call__(
        self,
        data: Dict = None,
        metadata: Dict = None,
        verbose: bool = True,
    ):
        data = data if data else {}
        metadata = metadata if metadata else {}
        output_data = {}

        t = c.timer()
        success = False

        fn = data['fn']
        kwargs = data.get('kwargs', {})
        args = data.get('args', [])
        user = data.get('user', [])
        try:
            self.check_call(fn=fn, args=args, kwargs=kwargs, user=user)
            c.print('Calling Function: ' + fn, color='cyan')
            output_data = getattr(self.module, fn)(*args, **kwargs)
            success = True
        except RuntimeError as ex:
            c.print(f'Exception in server: {ex}', color='red')
            if "There is no current event loop in thread" in str(ex):
                if verbose:
                    c.print('SETTING NEW ASYNCIO LOOP', color='yellow')
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
                return self.__call__(data=data, metadata=metadata)
        except Exception as ex:
            output_data = str(ex)
            if any([rex in output_data for rex in self.exceptions_to_raise]):
                raise ex
                self.stop()
            if verbose:
                c.print('[bold]EXCEPTION[/bold]: {ex}', color='red')

        sample_info = {
            'latency': t.seconds,
            'in_bytes': sys.getsizeof(data),
            'out_bytes': sys.getsizeof(output_data),
            'user': data.get('user', {}),
            'fn': fn,
            'timestamp': c.time(),
            'success': success,
        }

        # Calculate bps (bytes per second) for upload and download
        sample_info['upload_bps'] = sample_info['in_bytes'] / sample_info['latency']
        sample_info['download_bps'] = sample_info['out_bytes'] / sample_info['latency']

        c.print(sample_info)
        self.log_sample(sample_info)

        return {'data': {'result': output_data, 'info': sample_info}, 'metadata': metadata}

    def log_sample(self, sample_info: Dict, max_history: int = 100) -> None:
        if not hasattr(self, 'stats'):
            self.stats = {}

        sample_info['success'] = True

        self.stats['successes'] = self.stats.get('success', 0) + (1 if sample_info['success'] else 0)
        self.stats['errors'] = self.stats.get('errors', 0) + (1 if not sample_info['success'] else 0)
        self.stats['requests'] = self.stats.get('requests', 0) + 1
        self.stats['history'] = self.stats.get('history', []) + [sample_info]
        self.stats['most_recent'] = sample_info

        if len(self.stats['history']) > max_history:
            self.stats['history'].pop(0)

    def check_call(self, fn: str, args: List, kwargs: Dict, user: Dict):
        passed = False

        if fn == 'getattr':
            if len(args) == 1:
                attribute = args[0]
            elif 'k' in kwargs:
                attribute = kwargs['k']
            else:
                raise Exception('You are calling {k} which is an invalid attribute')

            # Is it an allowed attribute?
            if attribute in self.allowed_attributes:
                passed = True
        else:
            if fn in self.allowed_functions:
                passed = True

        return {'passed': passed}
    def set_serve(self, ip:str, port:int):
        self.server = HTTPServer((self.ip, self.port), self.HandlerClass)
        self.server.timeout = self.timeout
    def start(self) -> 'Server':
        r"""Starts the HTTP server.
        """
        try:
            logger.success("Server Started:".ljust(20) + "<blue>{}</blue>", self.ip + ':' + str(self.port))
            self.server.serve_forever()
        except KeyboardInterrupt:
            logger.success("Server Stopped:".ljust(20) + "<blue>{}</blue>", self.ip + ':' + str(self.port))

    def stop(self) -> 'Server':
        r"""Stop the HTTP server.
        """
        # Stop the server here if needed
        return self

    def start_server(cls):
        module = c.module('module')()  # Replace with your actual module or class
        server = Server(module=module, ip='0.0.0.0', port=8080)
        server.start()


    def test(self):
        # Define your test data
        test_data = {
            'fn': 'your_function_name',
            'args': [1,2,],
            'kwargs': {'key1': 1, 'key2': 2},
        }

        # Make the test request
        response = self.__call__(data=test_data, metadata={})

        # Process the response
        result = response['data']['result']
        info = response['data']['info']
        success = info['success']
        latency = info['latency']
        upload_bps = info['upload_bps']
        download_bps = info['download_bps']

        # Perform your assertions or print the results
        if success:
            print("Test passed!")
            print("Result:", result)
            print("Latency:", latency)
            print("Upload BPS:", upload_bps)
            print("Download BPS:", download_bps)
        else:
            print("Test failed!")




