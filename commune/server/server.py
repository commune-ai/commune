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
            find_port: bool = True, # find an existing port
            replace_port: bool = False,
            max_workers: Optional[int] = None, 
            maximum_concurrent_rpcs: Optional[int] = None,
            blacklist: Optional['Callable'] = None,
            thread_pool: Optional[futures.ThreadPoolExecutor] = None,
            timeout: Optional[int] = None,
            compression:Optional[str] = None,
            server: Optional['grpc._Server'] = None,
            config: Optional['commune.config'] = None,
            verbose: bool = True,
            whitelist_functions: Optional[List[str]] = ['functions', 'function_schema_map', 'getattr', 'servers', 'external_ip', 'pm2_status', 'peer_registry'],

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
          
        """ 
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            
        # 
        config = copy.deepcopy(config if config else self.default_config())
        
        
        self.max_workers = config.max_workers = max_workers if max_workers != None else config.max_workers
        self.maximum_concurrent_rpcs  = config.maximum_concurrent_rpcs = maximum_concurrent_rpcs if maximum_concurrent_rpcs != None else config.maximum_concurrent_rpcs
        self.compression = config.compression = compression if compression != None else config.compression
        self.timeout = timeout if timeout else config.timeout
        self.verbose = verbose
        
        self.check_config( config )
        self.config = config
        
        ip = ip if ip else self.default_ip
        self.external_ip = commune.external_ip()
        
        self.set_server( ip=ip, port=port, thread_pool=thread_pool, max_workers=max_workers, server=server) 


        # set the module
        self.module = module
        
        # whether or not the server is running
        self.started = False
        
        self.init_stats()
        
        # set the whitelist functions
        self.whitelist_functions = whitelist_functions + self.module.functions()
    def add_whitelist_functions(self, functions: List[str]):
        self.whitelist_functions += functions
    def add_blacklist_functions(self, functions: List[str]):
        self.blacklist_functions += functions
        
    def set_thread_pool(self, thread_pool: 'ThreadPoolExecutor' = None, max_workers: int = 10) -> 'ThreadPoolExecutor':
        if thread_pool == None:
            thread_pool = futures.ThreadPoolExecutor(max_workers=max_workers)
        
        self.thread_pool = thread_pool
        return thread_pool
    
    
    def set_server(self,  ip: str=  None , port:int =  None, 
                   thread_pool: 'ThreadPoolExecutor' = None, max_workers:int = 1, 
                   find_port:bool = True, replace_port:bool=True, 
                   server:'Server' = None ) -> 'Server':
        
        port = port if port != None else self.config.port
        ip = ip if ip != None else self.config.ip
        
        is_port_available = self.port_available(ip=ip, port=port)
        
        while not is_port_available:
            if find_port:
                port = self.get_available_port(ip=ip)
            if replace_port:
                self.kill_port(port=port)
            is_port_available =  self.port_available(ip=ip, port=port)

        

        is_port_available =  self.port_available(ip=ip, port=port)
        
        self.thread_pool = self.set_thread_pool(thread_pool=thread_pool)
        

        server = grpc.server( self.thread_pool,
                            #   interceptors=(ServerInterceptor(blacklist=blacklist,receiver_hotkey=self.wallet.hotkey.ss58_address),),
                                maximum_concurrent_rpcs = self.config.maximum_concurrent_rpcs,
                                options = [('grpc.keepalive_time_ms', 100000),
                                            ('grpc.keepalive_timeout_ms', 500000)]
                            )
    
    
        self.ip = ip
        self.port = port
        
        # set the server compression algorithm
        self.server = server
        commune.server.grpc.add_ServerServicer_to_server( self, server )
        self.full_address = str( ip ) + ":" + str( port )
        self.server.add_insecure_port( self.full_address )
        return self.server
    
    @classmethod   
    def help(cls):
        """ Print help to stdout
        """
        parser = argparse.ArgumentParser()
        cls.add_args( parser )
        print (cls.__new__.__doc__)
        parser.print_help()
    @classmethod
    def default_config(cls):
        config = commune.config()
        config.port = cls.get_available_port()
        config.ip =  '0.0.0.0'
        config.max_workers = 10
        config.maximum_concurrent_rpcs =  400
        config.compression = 'NoCompression'
        config.timeout = 10
        return config

    @classmethod   
    def check_config(cls, config: 'commune.config' ):
        """ Check config for axon port and wallet
        """
        assert config.port > 1024 and config.port < 65535, 'port must be in range [1024, 65535]'
        assert config.ip != None, 'ip must be set'


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
         
    def __call__(self, data:dict = None, metadata:dict = None):
        data = data if data else {}
        metadata = metadata if metadata else {}
        output_data = {}

        try:
            if 'fn' in data:
                fn_kwargs = data.get('kwargs', {})
                fn_args = data.get('args', [])
                
                assert data['fn'] in self.whitelist_functions, f'Function {data["fn"]} not in whitelist'
                
                commune.print('Calling Function: '+data['fn'], color='cyan')
            
                output_data = getattr(self.module, data['fn'])(*fn_args,**fn_kwargs)
            else:
                if hasattr(self.module, 'forward'):
                    data = self.module.forward(**data)
                elif hasattr(self.module, '__call__'):
                    data = self.module.__call__(**data)
                else:
                    raise Exception('module should have forward or __call__ for its default response')
        except RuntimeError as ex:
            commune.print(f'Exception in server: {ex}', 'red')
            if "There is no current event loop in thread" in str(ex):
                commune.print(f'Setting new loop', 'yellow')
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
                return self.__call__(data=data, metadata=metadata)
            
        except Exception as ex:
            commune.print(f'Exception in server: {ex}', 'red')
            output_data = str(ex)
            commune.log(output_data, 'red')
        
        # calculate states
        self.stats['call_count'] += 1
        # self.stats['in_bytes'] += sys.getsizeof(data)
        # self.stats['out_bytes'] += sys.getsizeof(output_data)
        # self.stats['in_bytes_per_call'] = self.stats['in_bytes']/(self.stats['call_count'] + 1e-10)
        # self.stats['out_bytes_per_call'] = self.stats['out_bytes']/(self.stats['call_count']+ 1e-10)
        
        
        return {'data': {'result': output_data}, 'metadata': metadata}
    

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




        
        t = commune.timer()
        request = self.deserialize(request)
        self.stats['time']['deserialize'] = t.seconds
        
        t = commune.timer()
        
        response = self(**request)
        self.stats['time']['module'] = t.seconds
        
        t = commune.timer()
        
        response = self.serialize(**response)
        self.stats['time']['serialize'] = t.seconds
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
        return f'{self.external_ip}:{self.port}'
    
    
    
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
    def get_available_port(cls, port_range: List[int] = None , ip:str =None) -> int:
        
        '''
        
        Get an available port within the {port_range} [start_port, end_poort] and {ip}
        '''
        port_range = port_range if port_range else cls.port_range
        ip = ip if ip else cls.default_ip
        
        # return only when the port is available
        for port in range(*port_range): 
            if cls.port_available(port=port, ip=ip):
                return port
    
        raise Exception(f'ports {port_range[0]} to {port_range[1]} are occupied, change the port_range to encompase more ports')

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
    def get_available_ports(cls, port_range: List[int] = None , ip:str =None) -> int:
        port_range = port_range if port_range else cls.port_range
        ip = ip if ip else cls.default_ip
        
        available_ports = []
        # return only when the port is available
        for port in range(*port_range): 
            if cls.port_available(port=port, ip=ip):
                available_ports.append(port)
        return available_ports
    
    @classmethod
    def port_available(cls,  port:int, ip:str = None):
        '''
        checks if a port is available
        '''
        
        import socket
        ip = ip if ip else cls.default_ip
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((ip, port))
        sock.close()
        # 0 when open, 111 otherwise
        return result != 0

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
    