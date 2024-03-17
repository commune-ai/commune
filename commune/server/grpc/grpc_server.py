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
import commune as c
from commune..server.grpc.interceptor import ServerInterceptor
from commune..server.grpc.serializer import Serializer
from commune..server.grpc.proto import ServerServicer
from commune..server.grpc.proto import DataBlock
import signal
from munch import Munch


class Server(ServerServicer, Serializer, c.Module):
    """ The factory class for commune.Server object
    The Server is a grpc server for the commune network which opens up communication between it and other neurons.
    The server protocol is defined in commune.proto and describes the manner in which forward and backwards requests
    are transported / encoded between validators and servers
    """
    port_range = [50050, 50100]
    default_ip =  '0.0.0.0'

    def __init__(
            self,
            module: Union[c.Module, object]= None,
            name = None,
            ip: Optional[str] = None,
            port: Optional[int] = None,
            max_workers: Optional[int] = 10, 
            authenticate = False,
            maximum_concurrent_rpcs: Optional[int] = 400,
            thread_pool: Optional[futures.ThreadPoolExecutor] = None,
            timeout: Optional[int] = None,
            compression:Optional[str] = None,
            server: Optional['grpc._Server'] = None,
            verbose: bool = True,
            whitelist: List[str] = None,
            blacklist: List[str ] = None,
            loop: 'asyncio.Loop' = None,
            exceptions_to_raise = ['CUDA out of memory',  'PYTORCH_CUDA_ALLOC_CONF'],
            network: 'Network' = None,


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



        self.set_event_loop(loop)
        
        
        if name == None:
            if not hasattr(module, 'module_name'):
                name = str(module)
        self.name = name
        self.timeout = timeout
        self.verbose = verbose
        self.module = module
        self.authenticate = authenticate
        
        
        self.ip = ip = ip if ip != None else self.default_ip
        self.port = port = c.resolve_port(port)
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
        from commune..server.grpc.proto import add_ServerServicer_to_server
        add_ServerServicer_to_server( self, server )
        self.full_address = str( ip ) + ":" + str( port )
        self.server.add_insecure_port( self.full_address )
    
        self.ip = c.external_ip()
        self.port = port
        
        # whether or not the server is running
        self.started = False
        self.set_stats()

        self.exceptions_to_raise = exceptions_to_raise
        self.set_module_info()
    def set_event_loop(self, loop: 'asyncio.AbstractEventLoop' = None) -> None:
        if loop == None:
            loop = c.get_event_loop()
        self.loop = loop
    
    def set_whitelist(self, functions: List[str]):
        if functions == None:
            functions = []
        self.whitelist = list(set(functions + c.helper_functions))
        
    def set_module_info(self):
        self.module_info = self.module.info()
        self.allowed_functions = self.module_info['functions']
        self.allowed_attributes = self.module_info['attributes']
        
        
    def check_call(self, fn:str, args:list, kwargs:dict, user:dict):
        passed = False

        if fn == 'getattr':
            if len(args) == 1:
                attribute = args[0]
            elif 'k' in kwargs:
                attribute = kwargs['k']
            else:
                raise Exception('you are falling {k} which is an attribute that is invalid')

            # is it an allowed attribute
            if attribute in self.allowed_attributes:
                passed = True
        else:

            if  fn in self.allowed_functions:
                passed = True
        

            
                
            
            
            
        {'passed': False}
        
    def set_thread_pool(self, thread_pool: 'ThreadPoolExecutor' = None, max_workers: int = 10) -> 'ThreadPoolExecutor':
        if thread_pool == None:
            thread_pool = futures.ThreadPoolExecutor(max_workers=max_workers)
        
        self.thread_pool = thread_pool
        return thread_pool
    

        

    
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
    
   
   
    def set_stats(self):
        self.stats = dict(
            call_count = 0,
            total_bytes = 0,
            time = {}
        )
        
    def __call__(self,
                 data:dict = None, 
                 metadata:dict = None,
                 verbose: bool = True,):
        data = data if data else {}
        metadata = metadata if metadata else {}
        output_data = {}
        
        
        t = c.timer()
        success = False
        
        fn = data['fn']
        kwargs = data.get('kwargs', {})
        args = data.get('args', [])
        user = data.get('user', [])
        auth = data.get('auth', None)
        if auth != None:
            c.print(auth)
        try:
            
            self.check_call(fn=fn, args=args, kwargs=kwargs, user=user)

            c.print('Calling Function: '+fn, color='cyan')
            fn_obj = getattr(self.module, fn)
            
            if callable(fn_obj):
                # if the function is callable (an actual function)
                output_data = fn_obj(*args, **kwargs)
            else:
                # if the function is an attribute
                output_data = fn_obj
            
            success = True

        except RuntimeError as ex:
            c.print(f'Exception in server: {ex}', color= 'red')
            if "There is no current event loop in thread" in str(ex):
                if verbose:
                    c.print(f'SETTING NEW ANSYNCIO LOOP', color='yellow')
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
                return self.__call__(data=data, metadata=metadata)
            
        except Exception as ex:
            output_data = str(ex)
            if any([rex in output_data for rex in self.exceptions_to_raise]):
                raise(ex)
                self.stop()
            
            if verbose:
                c.print(f'[bold]EXCEPTION[/bold]: {ex}', color='red')
        

        sample_info ={
            'latency': t.seconds,
            'in_bytes': sys.getsizeof(data),
            'out_bytes': sys.getsizeof(output_data),
            'user': data.get('user', {}),
            'fn': fn,
            'timestamp': c.time(),
            'success': success
            }
        
        # calculate bps (bytes per second) for upload and download
        sample_info['upload_bps'] = sample_info['in_bytes'] / sample_info['latency']
        sample_info['download_bps'] = sample_info['out_bytes'] / sample_info['latency']
        
        c.print(sample_info)
        self.log_sample(sample_info)
        
        

        return {'data': {'result': output_data, 'info': sample_info }, 'metadata': metadata}
    

    def log_sample(self, sample_info: dict, max_history: int = 100) -> None:
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
            
    def Forward(self, request: DataBlock, context: grpc.ServicerContext) -> DataBlock:
        r""" The function called by remote GRPC Forward requests. The Datablock is a generic formatter.
            
            Args:
                request (:obj:`DataBlock`, `required`): 
                    Tensor request proto.
                context (:obj:`grpc.ServicerContext`, `required`): 
                    grpc server context.
            
            Returns:
                response (c.proto.DataBlock): 
                    proto response carring the nucleus forward output or None under failure.
        """




        
        deserialize_timer = c.timer()
        request = self.deserialize(request)
        self.stats['time']['deserialize'] = deserialize_timer.seconds
        
        forward_timer = c.timer()
        response = self(**request)
        self.stats['time']['module'] = forward_timer.seconds
        
        serializer_timer = c.timer()
        response = self.serialize(**response)
        self.stats['time']['serialize'] = serializer_timer.seconds
        return response

    def __del__(self):
        r""" Called when this axon is deleted, ensures background threads shut down properly.
        """
        c.print('deregister')
        c.deregister_server(name=self.name)
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

    @property
    def address(self):
        return f'{self.ip}:{self.port}'
    
    
    
    def serve(self,
              wait_for_termination:bool=False,
              update_period:int = 10, 
              verbose:bool= True,
              register:bool=True):
        '''
        Serve the server and loop it until termination.
        '''
        self.start(wait_for_termination=False)

        lifetime_seconds:int = 0
        
        def print_serve_status():
            text = f'{str(self.module.server_name)} IP::{self.endpoint} LIFETIME(s): {lifetime_seconds}s'
            c.print(text, color='green')
            
        
        # register the server
        if register:
            c.register_server(name=self.name, 
                                          ip=self.ip,
                                          port=self.port)

 
        
        
        try:
            while True:
                if not wait_for_termination:
                    break
                lifetime_seconds += update_period
                if verbose:
                    print_serve_status()
                    
                    time.sleep(update_period)
                    
        except Exception as e:
            c.deregister_server(name=self.name)
            raise e
        
        

                
            



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
        c.deregister_server(name=self.name)
        if self.server != None:
            print('stopping server', self.server.__dict__)
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

        return not c.port_used(port=port, ip=ip)

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
        
        
        c.Client()
        module.stop()


    @property
    def info(self):
        '''
        Any server info
        '''
        return dict(
            ip=self.ip,
            port= self.port,
            address = self.endpoint,
            whitelist=self.whitelist,
            blacklist=self.blacklist,
        )
        
        

    