import argparse
import os
import copy
import inspect
import time
from concurrent import futures
from typing import Dict, List, Callable, Optional, Tuple, Union
import sys
import torch
import grpc
from substrateinterface import Keypair
from loguru import logger
import sys
import os
import asyncio
import commune
from commune.server.interceptor import ServerInterceptor
from commune.server.serializer import Serializer
from commune.server.proto import ServerServicer
from commune.server.proto import DataBlock
import signal

if os.getenv('USE_STREAMLIT'):
    import streamlit as st
from munch import Munch


class Server(ServerServicer, Serializer):
    """ The factory class for commune.Server object
    The Server is a grpc server for the commune network which opens up communication between it and other neurons.
    The server protocol is defined in commune.proto and describes the manner in which forward and backwards requests
    are transported / encoded between validators and servers
    """
    port_range = [50050, 50100]
    default_ip =  '0.0.0.0'

    def __init__(
            self,
            module: Union['Module', object]= None,
            ip: Optional[str] = None,
            port: Optional[int] = None,
            max_workers: Optional[int] = 10, 
            maximum_concurrent_rpcs: Optional[int] = 400,
            authenticate: bool = False,
            thread_pool: Optional[futures.ThreadPoolExecutor] = None,
            timeout: Optional[int] = None,
            compression:Optional[str] = None,
            server: Optional['grpc._Server'] = None,
            verbose: bool = True,
            whitelist_functions: List[str] = [],
            blacklist_functions: List[str ] = [],
            loop: 'AscynioLoop' = None,
            subspace = None,


        ) -> 'Server':
        r""" Creates a new commune.Server object from passed arguments.
            Args:
                thread_pool (:obj:`Optional[ThreadPoolExecutor]`, `optional`):
                    Threadpool used for processing server queries.
                server (:obj:`Optional[grpc._Server]`, `required`):
                    Grpc server endpoint, overrides passed threadpool.
                port (:type:`Optional[int]`, `optional`):
                    Binding port.
                ip (:type:`Optional[str]`, `optional`):
                    Binding ip.
                external_ip (:type:`Optional[str]`, `optional`):
                    The external ip of the server to broadcast to the network.
                max_workers (:type:`Optional[int]`, `optional`):
                    Used to create the threadpool if not passed, specifies the number of active threads servicing requests.
                maximum_concurrent_rpcs (:type:`Optional[int]`, `optional`):
                    Maximum allowed concurrently processed RPCs.
                timeout (:type:`Optional[int]`, `optional`):
                    timeout on the forward requests. 
                authenticate (:type:`Optional[bool]`, `optional`):
                    Whether or not to authenticate the server.
          
        """ 



        self.set_event_loop(loop=loop)

        self.set_server( ip=ip, 
                        port=port, 
                        thread_pool=thread_pool,
                        max_workers=max_workers,
                        maximum_concurrent_rpcs=maximum_concurrent_rpcs,
                        compression=compression,) 
        
        self.timeout = timeout
        self.verbose = verbose
        
        # set the module
        self.module = module
        
        # set the whitelist functions
        self.add_whitelist_functions(whitelist_functions + self.module.functions())
        self.add_blacklist_functions(blacklist_functions)
        self.authenticate = authenticate
        self.subspace = subspace
        
        
    
    
    def set_event_loop(self, loop: 'asyncio.AbstractEventLoop' = None) -> None:
        if loop == None:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        self.loop = loop
    
    def add_whitelist_functions(self, functions: List[str]):
        if not hasattr(self, 'whitelist_functions'):
            self.whitelist_functions = []
            
        commune.print(f'Adding whitelist functions: {functions}','purple')
        self.whitelist_functions += functions
    def add_blacklist_functions(self, functions: List[str]):
        if not hasattr(self, 'blacklist_functions'):
            self.blacklist_functions = []
        self.blacklist_functions += functions
        
    def set_thread_pool(self, thread_pool: 'ThreadPoolExecutor' = None, max_workers: int = 10) -> 'ThreadPoolExecutor':
        if thread_pool == None:
            thread_pool = futures.ThreadPoolExecutor(max_workers=max_workers)
        
        self.thread_pool = thread_pool
        return thread_pool
    
    
    def set_server(self, 
                   ip: str=  None ,
                   port:int =  None, 
                   thread_pool: 'ThreadPoolExecutor' = None,
                   max_workers:int = 1 ,
                   maximum_concurrent_rpcs: int = 400,
                   compression: str  = '' ) -> 'Server':
        
        ip = ip if ip != None else self.default_ip
        port = commune.resolve_port(port)
        while not self.port_available(ip=ip, port=port):
            port = self.get_available_port(ip=ip)
            is_port_available =  self.port_available(ip=ip, port=port)
        
        self.thread_pool = self.set_thread_pool(thread_pool=thread_pool)
        

        server = grpc.server( self.thread_pool,
                            #   interceptors=(ServerInterceptor(blacklist=blacklist,receiver_hotkey=self.wallet.hotkey.ss58_address),),
                                maximum_concurrent_rpcs = maximum_concurrent_rpcs,
                                options = [('grpc.keepalive_time_ms', 100000),
                                            ('grpc.keepalive_timeout_ms', 500000)]
                            )
        
        # set the server compression algorithm
        self.server = server
        commune.server.grpc.add_ServerServicer_to_server( self, server )
        self.full_address = str( ip ) + ":" + str( port )
        self.server.add_insecure_port( self.full_address )
    
        self.ip = commune.external_ip()
        self.port = port
        
        # whether or not the server is running
        self.started = False
        self.init_stats()
        

        return self.server
    
    @classmethod   
    def help(cls):
        """ Print help to stdout
        """
        parser = argparse.ArgumentParser()
        cls.add_args( parser )
        print (cls.__new__.__doc__)
        parser.print_help()


    def __str__(self) -> str:
        return "Server({}, {}, {})".format( self.ip, self.port,  "started" if self.started else "stopped")

    def __repr__(self) -> str:
        return self.__str__()
    
   
   
    def init_stats(self):
        self.stats = dict(
            call_count = 0,
            total_bytes = 0,
            time = {}
        )
        
    def resolve_authentication(self, data: dict = None, metadata: dict = None):
        
        if self.authenticate:
            auth = data.pop('auth', None)
            assert isinstance(auth, dict), 'Please provide authentication Old Chap' 
            assert hasattr(self.module, 'authenticate')
            assert self.module.authenticate(auth=auth)
        
        return data, metadata
    def __call__(self,
                 data:dict = None, 
                 metadata:dict = None,
                 verbose: bool = True,):
        data = data if data else {}
        metadata = metadata if metadata else {}
        output_data = {}
        
        
        t = commune.timer()
        
        try:
            
            data, metadata = self.resolve_authentication(data=data, metadata=metadata)

            fn = data['fn']
            fn_kwargs = data.get('kwargs', {})
            fn_args = data.get('args', [])
            
            assert fn in self.whitelist_functions, f'Function {data["fn"]} not in whitelist'
            assert fn not in self.blacklist_functions, f'Function {data["fn"]} in blacklist'
            
            if verbose:
                commune.print('Calling Function: '+fn, color='cyan')
            output_data = getattr(self.module, fn)(*fn_args,**fn_kwargs)

        except RuntimeError as ex:
            commune.print(f'Exception in server: {ex}', 'red')
            if "There is no current event loop in thread" in str(ex):
                commune.print(f'Setting new loop', 'yellow')
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
                return self.__call__(data=data, metadata=metadata)
            
        except Exception as ex:
            commune.print(f'Exception in server: {ex}', 'red')
        
        sample_info ={
            'seconds': t.seconds,
            'in_bytes': sys.getsizeof(data),
            'out_bytes': sys.getsizeof(output_data),
            'auth': data.get('auth', None),
            'fn': fn,
            'timestamp': commune.time()
            }
        
        

        return {'data': {'result': output_data, 'info': sample_info }, 'metadata': metadata}
    

    def log_sample(sample_info: dict, max_history: int = None) -> None:
        
            max_history = max_history if max_history else 100
            if not hasattr(self, 'sample_info_history'):
                self.sample_info_history = []
            
            self.sample_info_history.append(sample_info)
            if len(sample_info_history) > max_history:
                self.sample_info_history.pop(0)
            # calculate states
            self.stats['count'] += 1
            for k,v in sample_info.items():
                if type(v) in [int, float]:
                    prev_v = self.stats.get(k, 0)
                    self.stats[k] = (v + prev_v*(self.stats['count'] - 1)) / self.stats['call_count']

    def Forward(self, request: DataBlock, context: grpc.ServicerContext) -> DataBlock:
        r""" The function called by remote GRPC Forward requests. The Datablock is a generic formatter.
            
            Args:
                request (:obj:`DataBlock`, `required`): 
                    Tensor request proto.
                context (:obj:`grpc.ServicerContext`, `required`): 
                    grpc server context.
            
            Returns:
                response (commune.proto.DataBlock): 
                    proto response carring the nucleus forward output or None under failure.
        """




        
        deserialize_timer = commune.timer()
        request = self.deserialize(request)
        self.stats['time']['deserialize'] = deserialize_timer.seconds
        
        forward_timer = commune.timer()
        response = self(**request)
        self.stats['time']['module'] = forward_timer.seconds
        
        serializer_timer = commune.timer()
        response = self.serialize(**response)
        self.stats['time']['serialize'] = serializer_timer.seconds
        return response

    def __del__(self):
        r""" Called when this axon is deleted, ensures background threads shut down properly.
        """
        if hasattr(self, 'server'):
            self.stop()

    @property
    def id(self) -> str:
        return f'{self.__class__.name}(endpoint={self.endpoint}, model={self.model_name})'


    @classmethod
    def argparse(cls):
        parser = argparse.ArgumentParser(description='Gradio API and Functions')
        parser.add_argument('-fn', '--function', dest='function', help='run a function from the module', type=str, default="streamlit")
        parser.add_argument('-kwargs', '--kwargs', dest='kwargs', help='arguments to the function', type=str, default="{}")  
        parser.add_argument('-args', '--args', dest='args', help='arguments to the function', type=str, default="[]")  
        return parser.parse_args()


    @classmethod
    def run(cls): 
        input_args = cls.argparse()
        assert hasattr(cls, input_args.function)
        kwargs = json.loads(input_args.kwargs)
        assert isinstance(kwargs, dict)
        
        args = json.loads(input_args.args)
        assert isinstance(args, list)
        getattr(cls, input_args.function)(*args, **kwargs)
        
    @property
    def endpoint(self):
        return f'{self.ip}:{self.port}'
    
    
    
    def serve(self,
              wait_for_termination:bool=False,
              update_period:int = 10, 
              verbose:bool= True):
        '''
        Serve the server and loop it until termination.
        '''
        self.start(wait_for_termination=False)

        lifetime_seconds:int = 0
        
        def print_serve_status():
            text = f'Serving {str(self.module.module_id)} IP::{self.endpoint} LIFETIME(s): {lifetime_seconds}s STATE: {dict(self.stats)}'
            commune.print(text=text, color='green')


        while True:
            if not wait_for_termination:
                break
            lifetime_seconds += update_period
            if verbose:
                print_serve_status()
                time.sleep(update_period)

                
            



    def start(self, wait_for_termination=False) -> 'Server':
        r""" Starts the standalone axon GRPC server thread.
        """
        if self.server != None:
            self.server.stop( grace = 1 )  
            logger.success("Server Stopped:".ljust(20) + "<blue>{}</blue>", self.ip + ':' + str(self.port))

        self.server.start()
        logger.success("Server Started:".ljust(20) + "<blue>{}</blue>", self.ip + ':' + str(self.port))
        self.started = True
        if wait_for_termination:
            self.server.wait_for_termination()

        return self

    def stop(self) -> 'Server':
        r""" Stop the axon grpc server.
        """
        if self.server != None:
            self.server.stop( grace = 1 )
            logger.success("Server Stopped:".ljust(20) + "<blue>{}</blue>", self.ip + ':' + str(self.port))
        self.started = False

        return self

    @staticmethod
    def kill_port(port:int)-> str:
        from psutil import process_iter
        '''
        Kills the port {port}
        '''
        for proc in process_iter():
            for conns in proc.connections(kind='inet'):
                if conns.laddr.port == port:
                    proc.send_signal(signal.SIGKILL) # or SIGKILL
        return port


    @classmethod
    def get_used_ports(cls, port_range: List[int] = None , ip:str =None) -> int:
        port_range = port_range if port_range else cls.port_range
        ip = ip if ip else cls.default_ip
        used_ports = []
        # return only when the port is available
        for port in range(*port_range): 
            if not cls.port_available(port=port, ip=ip):
                used_ports.append(port)
        return used_ports
    


    @classmethod
    def port_available(cls,  port:int, ip:str = None):
        '''
        checks if a port is available
        '''

        return not commune.port_used(port=port, ip=ip)

    @classmethod
    def test_server(cls):
        
        class DemoModule:
            def __call__(self, data:dict, metadata:dict) -> dict:
                return {'data': data, 'metadata': {}}
        
        modules = {}
        for m in range(10):
            module = Server(module=DemoModule())
            # module.start()
            modules[module.port] = module
        
        
        commune.Client()
        module.stop()


    @property
    def info(self):
        '''
        Any server info
        '''
        return dict(
            ip=self.ip,
            port= self.port,
        )
        
        

if __name__ == '__main__':
    import asyncio 
    import random
    import streamlit as st
    Server.test_server()
    