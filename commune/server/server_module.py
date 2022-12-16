import argparse
import os
import copy
import inspect
import time
from concurrent import futures
from typing import Dict, List, Callable, Optional, Tuple, Union
import streamlit as st
import sys
import torch
import grpc
from substrateinterface import Keypair
from loguru import logger
import sys
import os
import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())
sys.path.append(os.getenv('PWD'))
import commune
from commune.server.server_interceptor import ServerInterceptor
from commune.serializer import SerializerModule
from commune.proto import CommuneServicer
import bittensor





class ServerModule(CommuneServicer, SerializerModule):
    """ The factory class for bittensor.Axon object
    The Axon is a grpc server for the bittensor network which opens up communication between it and other neurons.
    The server protocol is defined in bittensor.proto and describes the manner in which forward and backwards requests
    are transported / encoded between validators and servers
    """

    def __init__(
            self,
            config: Optional['commune.Config'] = None,
            wallet: Optional['bittensor.Wallet'] = None,
            server: Optional['grpc._Server'] = None,
            port: Optional[int] = None,
            ip: Optional[str] = None,
            module: 'AxonModule'= None,
            serializer: 'SerializerModule'= None,
            external_ip: Optional[str] = None,
            external_port: Optional[int] = None,
            max_workers: Optional[int] = None, 
            maximum_concurrent_rpcs: Optional[int] = None,
            blacklist: Optional['Callable'] = None,
            thread_pool: Optional[futures.ThreadPoolExecutor] = None,
            timeout: Optional[int] = None,
            compression:Optional[str] = None,

        ) -> 'bittensor.Axon':
        r""" Creates a new bittensor.Axon object from passed arguments.
            Args:
                config (:obj:`Optional[bittensor.Config]`, `optional`): 
                    bittensor.Server.config()
                wallet (:obj:`Optional[bittensor.Wallet]`, `optional`):
                    bittensor wallet with hotkey and coldkeypub.    
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

        config = copy.deepcopy(config if config else self.default_config())
        self.port = config.port = port if port != None else config.port
        self.ip = config.ip = ip if ip != None else config.ip
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
        self.server = server
        self.module = module

        commune.grpc.add_CommuneServicer_to_server( self, server )
        full_address = str( config.ip ) + ":" + str( config.port )
        self.server.add_insecure_port( full_address )
        self.check_config( config )
        self.config = config

        self.started = False


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
                help='''The local ip this axon binds to. ie. [::]''', default = bittensor.defaults.Server.ip)
            parser.add_argument('--' + prefix_str + 'external_port', type=int, required=False,
                    help='''The public port this axon broadcasts to the network. i.e. 8091''', default = bittensor.defaults.Server.external_port)
            parser.add_argument('--' + prefix_str + 'external_ip', type=str, required=False,
                help='''The external ip this axon broadcasts to the network to. ie. [::]''', default = bittensor.defaults.Server.external_ip)
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
        config.port = 8091
        config.ip =  '[::]'
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

    def Forward(self, request: commune.proto.DataBlock, context: grpc.ServicerContext) -> commune.proto.DataBlock:
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


        request = self.deserialize(request)
        response = self.module(**request)
        response = self.serialize(**response)
        
        return response

    def __del__(self):
        r""" Called when this axon is deleted, ensures background threads shut down properly.
        """
        self.stop()

    def serve( 
            self, 
            use_upnpc: bool = False, 
            subtensor: 'bittensor.Subtensor' = None,
            network: str = None,
            chain_endpoint: str = None,
            prompt: bool = False
        ) -> 'Axon':
        r""" Subscribes this Axon servicing endpoint to the passed network using it's wallet.
            Args:
                use_upnpc (:type:bool, `optional`): 
                    If true, serves the axon attempts port forward through your router before 
                    subscribing.
                subtensor (:obj:`bittensor.Subtensor`, `optional`): 
                    Chain connection through which to serve.
                network (default='local', type=str)
                    If subtensor is not set, uses this network flag to create the subtensor connection.
                chain_endpoint (default=None, type=str)
                    Overrides the network argument if not set.
                prompt (bool):
                    If true, the call waits for confirmation from the user before proceeding.

        """   
        if subtensor == None: subtensor = bittensor.subtensor( network = network, chain_endpoint = chain_endpoint) 
        serv_success = subtensor.serve_axon( axon = self, use_upnpc = use_upnpc, prompt = prompt )
        if not serv_success:
            raise RuntimeError('Failed to serve neuron.')
        return self

    def start(self) -> 'ServerModule':
        r""" Starts the standalone axon GRPC server thread.
        """
        st.write(self.__dict__)
        if self.server != None:
            self.server.stop( grace = 1 )  
            logger.success("Axon Stopped:".ljust(20) + "<blue>{}</blue>", self.ip + ':' + str(self.port))

        self.server.start()
        logger.success("Axon Started:".ljust(20) + "<blue>{}</blue>", self.ip + ':' + str(self.port))
        self.started = True

        return self

    def stop(self) -> 'ServerModule':
        r""" Stop the axon grpc server.
        """
        if self.server != None:
            self.server.stop( grace = 1 )
            logger.success("Axon Stopped:".ljust(20) + "<blue>{}</blue>", self.ip + ':' + str(self.port))
        self.started = False

        return self

class DemoModule:
    def __call__(self, data:dict, metadata:dict) -> dict:
        return {'data': data, 'metadata': {}}


if __name__ == '__main__':
    module = ServerModule(module=DemoModule())
    module.start()
    st.write(module)


    # request = module.serialize(data=torch.ones(10,10))
    # response = module.Forward(request=request, context=None)
    # st.write(response)