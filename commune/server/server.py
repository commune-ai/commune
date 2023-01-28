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
import bittensor
import signal

if os.getenv('USE_STREAMLIT'):
    import streamlit as st
from munch import Munch


class Server(ServerServicer, Serializer):
    """ The factory class for bittensor.Axon object
    The Axon is a grpc server for the bittensor network which opens up communication between it and other neurons.
    The server protocol is defined in bittensor.proto and describes the manner in which forward and backwards requests
    are transported / encoded between validators and servers
    """
    port_range = [50050, 50100]
    default_ip =  '0.0.0.0'

    def __init__(
            self,
            module: Union['Module', object]= None,
            ip: Optional[str] = '0.0.0.0',
            port: Optional[int] = None,
            find_port: bool = True, # find an existing port
            replace_port: bool = False,
            external_ip: Optional[str] = None,
            external_port: Optional[int] = None,
            max_workers: Optional[int] = None, 
            maximum_concurrent_rpcs: Optional[int] = None,
            blacklist: Optional['Callable'] = None,
            thread_pool: Optional[futures.ThreadPoolExecutor] = None,
            timeout: Optional[int] = None,
            compression:Optional[str] = None,
            serializer: 'Serializer'= None,
            server: Optional['grpc._Server'] = None,
            config: Optional['commune.Config'] = None,

        ) -> 'Server':
        r""" Creates a new bittensor.Axon object from passed arguments.
            Args:
                config (:obj:`Optional[bittensor.Config]`, `optional`): 
                    bittensor.Server.config()  
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
                external_port (:type:`Optional[int]`, `optional`):
                    The external port of the server to broadcast to the network.
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

        port = port if port != None else config.port
        ip = ip if ip != None else config.ip
        is_port_available =  self.port_available(ip=ip, port=port)
        
        while not is_port_available:
            if find_port:
                port = self.get_available_port(ip=ip)
            if replace_port:
                self.kill_port(port=port)
            is_port_available =  self.port_available(ip=ip, port=port)

        self.port = config.port = port
        self.ip = config.ip = ip
        
        self.external_ip = config.external_ip = external_ip if external_ip != None else config.external_ip
        self.external_port = config.external_port = external_port if external_port != None else config.external_port
        self.max_workers = config.max_workers = max_workers if max_workers != None else config.max_workers
        self.maximum_concurrent_rpcs  = config.maximum_concurrent_rpcs = maximum_concurrent_rpcs if maximum_concurrent_rpcs != None else config.maximum_concurrent_rpcs
        self.compression = config.compression = compression if compression != None else config.compression
        self.timeout = timeout if timeout else config.timeout


        # Determine the grpc compression algorithm
        if config.compression == 'gzip':
            compress_alg = grpc.Compression.Gzip
        elif config.compression == 'deflate':
            compress_alg = grpc.Compression.Deflate
        else:
            compress_alg = grpc.Compression.NoCompression
        
        
        if thread_pool == None:
            thread_pool = futures.ThreadPoolExecutor( max_workers = config.max_workers )

        if server == None:
            server = grpc.server( thread_pool,
                                #   interceptors=(ServerInterceptor(blacklist=blacklist,receiver_hotkey=self.wallet.hotkey.ss58_address),),
                                  maximum_concurrent_rpcs = config.maximum_concurrent_rpcs,
                                  options = [('grpc.keepalive_time_ms', 100000),
                                             ('grpc.keepalive_timeout_ms', 500000)]
                                )
        
        # set the server compression algorithm
        self.server = server
        commune.server.grpc.add_ServerServicer_to_server( self, server )
        full_address = str( config.ip ) + ":" + str( config.port )
        self.server.add_insecure_port( full_address )
        self.check_config( config )
        self.config = config

        # set the module
        self.module = module
        
        # whether or not the server is running
        self.started = False
        
        self.init_stats()

    @classmethod   
    def help(cls):
        """ Print help to stdout
        """
        parser = argparse.ArgumentParser()
        cls.add_args( parser )
        print (cls.__new__.__doc__)
        parser.print_help()

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser, prefix: str = None  ):
        """ Accept specific arguments from parser
        """
        prefix_str = '' if prefix == None else prefix + '.'
        try:
            parser.add_argument('--' + prefix_str + 'port', type=int, 
                    help='''The local port this axon endpoint is bound to. i.e. 8091''', default = bittensor.defaults.Server.port)
            parser.add_argument('--' + prefix_str + 'ip', type=str, 
                help='''The local ip this axon binds to. ie. 0.0.0.0''', default = bittensor.defaults.Server.ip)
            parser.add_argument('--' + prefix_str + 'external_port', type=int, required=False,
                    help='''The public port this axon broadcasts to the network. i.e. 8091''', default = bittensor.defaults.Server.external_port)
            parser.add_argument('--' + prefix_str + 'external_ip', type=str, required=False,
                help='''The external ip this axon broadcasts to the network to. ie. 0.0.0.0''', default = bittensor.defaults.Server.external_ip)
            parser.add_argument('--' + prefix_str + 'max_workers', type=int, 
                help='''The maximum number connection handler threads working simultaneously on this endpoint. 
                        The grpc server distributes new worker threads to service requests up to this number.''', default = bittensor.defaults.Server.max_workers)
            parser.add_argument('--' + prefix_str + 'maximum_concurrent_rpcs', type=int, 
                help='''Maximum number of allowed active connections''',  default = bittensor.defaults.Server.maximum_concurrent_rpcs)
            parser.add_argument('--' + prefix_str + 'compression', type=str, 
                help='''Which compression algorithm to use for compression (gzip, deflate, NoCompression) ''', default = bittensor.defaults.Server.compression)
        except argparse.ArgumentError:
            # re-parsing arguments.
            pass

    @classmethod
    def default_config(cls):
        config = commune.Config()
        config.port = cls.get_available_port()
        config.ip =  '0.0.0.0'
        config.external_port =  None
        config.external_ip =  None
        config.max_workers = 10
        config.maximum_concurrent_rpcs =  400
        config.compression = 'NoCompression'
        config.timeout = 10
        return config

    @classmethod   
    def check_config(cls, config: 'commune.Config' ):
        """ Check config for axon port and wallet
        """
        assert config.port > 1024 and config.port < 65535, 'port must be in range [1024, 65535]'
        assert config.external_port is None or (config.external_port > 1024 and config.external_port < 65535), 'external port must be in range [1024, 65535]'


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

        try:
            if 'fn' in data:
                fn_kwargs = data.get('kwargs', {})
                fn_args = data.get('args', [])
            
                data = getattr(self.module, data['fn'])(*fn_args,**fn_kwargs)
            else:
                # print(f'[green]{data}')
                if hasattr(self.module, 'forward'):
                    data = self.module.forward(**data)
                elif hasattr(self.module, '__call__'):
                    data = self.module.forward(**data)
                else:
                    raise Exception('module should have forward or __call__ for its default response')
        except RuntimeError as ex:
            if "There is no current event loop in thread" in str(ex):
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
                return self.__call__(data=data, metadata=metadata)
        self.stats['call_count'] += 1
        
        torch.cuda.empty_cache()

        return {'data': {'result': data}, 'metadata': metadata}
    
    # # DEFUALT FORWARD FUNCTION
    # def forward(**kwargs):
    #     return kwargs
    


    def Forward(self, request: DataBlock, context: grpc.ServicerContext) -> DataBlock:
        r""" The function called by remote GRPC Forward requests. The Datablock is a generic formatter.
            
            Args:
                request (:obj:`bittensor.proto`, `required`): 
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
        return f'{self.ip}:{self.port}'
    
    def serve(self,
              wait_for_termination:bool=False,
              update_period:int = 10, 
              verbose:bool= False):
        '''
        Serve the server and loop it until termination.
        '''
        self.start(wait_for_termination=False)

        lifetime_seconds:int = 0
        
        def print_serve_status():
            print(f'Serving {str(self.module.module_id)}::ip::{self.endpoint} LIFETIME(s): {lifetime_seconds}s STATE: {dict(self.stats)}')


        while True:
            print_serve_status()
            time.sleep(update_period)
            lifetime_seconds += update_period
            if not wait_for_termination:
                break
                
            



    def start(self, wait_for_termination=False) -> 'Server':
        r""" Starts the standalone axon GRPC server thread.
        """
        if self.server != None:
            self.server.stop( grace = 1 )  
            logger.success("Axon Stopped:".ljust(20) + "<blue>{}</blue>", self.ip + ':' + str(self.port))

        self.server.start()
        logger.success("Axon Started:".ljust(20) + "<blue>{}</blue>", self.ip + ':' + str(self.port))
        self.started = True
        if wait_for_termination:
            self.server.wait_for_termination()

        return self

    def stop(self) -> 'Server':
        r""" Stop the axon grpc server.
        """
        if self.server != None:
            self.server.stop( grace = 1 )
            logger.success("Axon Stopped:".ljust(20) + "<blue>{}</blue>", self.ip + ':' + str(self.port))
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
        print((ip, port))
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
        
        
        # print(module.started)
        
        # print(module.port_available(port=module.port - 1))

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
    