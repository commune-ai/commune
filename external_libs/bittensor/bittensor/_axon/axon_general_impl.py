""" Implementation of Axon, services Forward and Backward requests from other neurons.
"""
# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2022 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.

import sys
import time as clock
from types import SimpleNamespace
from typing import List, Tuple, Callable

import torch
import grpc
import wandb
import pandas
from loguru import logger
import torch.nn.functional as F
import concurrent

from prometheus_client import Counter, Histogram, Enum, CollectorRegistry

import bittensor
import bittensor.utils.stats as stat_utils
from datetime import datetime
from .axon_module_impl import AxonModule

logger = logger.opt(colors=True)

class AxonGeneral( bittensor.grpc.BittensorServicer ):
    r""" Services Forward and Backward requests from other neurons.
    """
    def __init__( 
        self, 
        wallet: 'bittensor.wallet',
        ip: str,
        port: int,
        external_ip: str,
        external_port: int,
        server: 'grpc._Server',
        module : AxonModule,
        timeout: int,
        priority_threadpool: 'bittensor.prioritythreadpool' = None,
    ):
        r""" Initializes a new Axon tensor processing endpoint.
            
            Args:
                config (:obj:`bittensor.Config`, `required`): 
                    bittensor.axon.config()
                wallet (:obj:`bittensor.wallet`, `required`):
                    bittensor wallet with hotkey and coldkeypub.
                server (:obj:`grpc._Server`, `required`):
                    Grpc server endpoint.
                module (:obj:list of `callable`, `optional`):
                    python class that is of type AxonModule.
                priority (:obj:`callable`, `optional`):
                    function to assign priority on requests.
                priority_threadpool (:obj:`bittensor.prioritythreadpool`, `optional`):
                    bittensor priority_threadpool.
        """
        self.ip = ip
        self.port = port
        self.external_ip = external_ip
        self.external_port = external_port
        self.wallet = wallet
        self.server = server
        self.module = module
        assert isinstance(self.module, AxonModule)

        self.timeout = timeout
        self.stats = self._init_stats()
        self.started = None
        

    def __str__(self) -> str:
        return "Axon({}, {}, {}, {})".format( self.ip, self.port, self.wallet.hotkey.ss58_address, "started" if self.started else "stopped")

    def __repr__(self) -> str:
        return self.__str__()

    def Forward(self, request: bittensor.proto.TensorMessage, context: grpc.ServicerContext) -> bittensor.proto.TensorMessage:
        r""" The function called by remote GRPC Forward requests from other neurons.
            Forward is equivalent to a 'forward' pass through a neural network.
            After checking request validity, this function passes the request to the nucleus for processing.
            See :obj:`bittensor.proto.ReturnCode` for all possible return codes.
            
            Args:
                request (:obj:`bittensor.proto`, `required`): 
                    Tensor request proto.
                context (:obj:`grpc.ServicerContext`, `required`): 
                    grpc server context.
            
            Returns:
                response (bittensor.proto.TensorMessage): 
                    proto response carring the nucleus forward output or None under failure.
        """
        forward_response_tensors, code, synapses = self.call_module( request )
        response = bittensor.proto.TensorMessage(
            version = bittensor.__version_as_int__, 
            hotkey = self.wallet.hotkey.ss58_address, 
            return_code = code,
            tensors = forward_response_tensors if forward_response_tensors is not None else [],
            requires_grad = request.requires_grad,
            synapses = synapses,
        )
        return response

    @staticmethod
    def check_if_should_return( synapse_codes:list) -> bool:
        '''
        Function which returns true if all codes are non success

        Returns:
            should_return (bool):
                should return
        '''
        should_return = True
        for code in synapse_codes:
            if code == bittensor.proto.ReturnCode.Success:
                should_return =  False
        return should_return

    def finalize_codes_stats_and_logs(self,  message = None):
        '''
        Function which prints all log statements per synapse
        '''
        for index, synapse in enumerate( synapses ):
            # === Logging
            request.synapses [ index ].return_code = synapse_codes[ index ] # Set synapse wire proto codes.
            request.synapses [ index ].message = synapse_messages[ index ] # Set synapse wire proto message
            bittensor.logging.rpc_log ( 
                axon = True, 
                forward = True, 
                is_response = synapse_is_response [index], 
                code = synapse_codes[ index ], 
                call_time = synapse_call_times[ index ], 
                pubkey = request.hotkey, 
                inputs = synapse_inputs [index] , 
                outputs = None if synapse_responses[index] == None else list( synapse_responses[index].shape ), 
                message = synapse_messages[ index ] if message == None else message,
                synapse = synapse.synapse_type
            )

    def call_module(self, request):
        r""" Performs validity checks on the grpc request before passing the tensors to the forward queue.
            Returns the outputs and synapses from the backend forward call.
            
            Args:
                request (:obj:`bittensor.proto`, `required`): 
                    Tensor request proto.
            Returns:
                response (:obj:`bittensor.proto.Tensor, `required`): 
                    serialized tensor response from the nucleus call or None.
                code (:obj:`bittensor.proto.ReturnCode`, `required`):
                    Code from the call. This specifies if the overall function call was a success. 
                    This is separate from the synapse returns codes which relate to the individual synapse call. 
                synapses (:obj:`List[ 'bittensor.proto.Synapse' ]` of shape :obj:`(num_synapses)`, `required`):
                    Synapse wire protos with return codes from forward request.
        """
        # ===================================================================
        # ==== First deserialize synapse wire protos to instance objects ====        
        # ===================================================================
        synapses: List['bittensor.Synapse'] = []
        for synapse_wire_proto in request.synapses:
            synapses.append( bittensor.synapse.deserialize( synapse_wire_proto ) )


        # ===================================
        # ==== Init params from synapses ====        
        # ===================================
        # These items are filled through the call and the function returns 
        # when all codes are non-success or the function finishes completely.
        synapse_messages = [ "Success" for _ in synapses ]
        synapse_codes = [ bittensor.proto.ReturnCode.Success for _ in synapses ]
        synapse_inputs = [ None for _ in synapses ]
        synapse_responses = [ synapse.empty() for synapse in synapses ] # We fill nones for non success.
        synapse_is_response = [ False for _ in synapses ]
        synapse_call_times = [ 0 for _ in synapses ]
        synapse_timeout = min( [self.synapse_timeouts[s.synapse_type] for s in synapses] + [bittensor.__blocktime__] )
        start_time = clock.time()

        # ======================================
        # ==== Check Empty request ====
        # ======================================
        if len(request.tensors) == 0:
            code = bittensor.proto.ReturnCode.EmptyRequest
            message = "Forward request contains {} tensors, expected 1 tensor in the forward call".format(len(request.tensors))
            call_time = clock.time() - start_time
            synapse_codes = [code for _ in synapses ]
            synapse_call_times = [call_time for _ in synapses ]
            synapse_messages = [ message for _ in synapses ]
            self.finalize_codes_stats_and_logs()
            return [], code, request.synapses

        
        # ======================================
        # ==== Check request length ====
        # ======================================
        if len( request.tensors ) != len( synapses ):
            # Not enough responses per request.
            code = bittensor.proto.ReturnCode.RequestShapeException
            call_time = clock.time() - start_time
            message = "Request length doesn't match synape length."
            synapse_codes = [code for _ in synapses ]
            synapse_call_times = [call_time for _ in synapses ]
            synapse_messages = [ message for _ in synapses ]
            self.finalize_codes_stats_and_logs()
            return [], bittensor.proto.ReturnCode.RequestShapeException, request.synapses


        # ===================================
        # ==== Deserialize/Check inputs ====
        # ===================================
        deserialized_forward_tensors = [ None for _ in synapses]
        for index, synapse in enumerate( synapses ):
            try:
                deserialized_forward_tensors [index] = synapse.deserialize_forward_request_tensor ( request.tensors [index] )

            except Exception as e:
                synapse_codes [index] = bittensor.proto.ReturnCode.RequestDeserializationException
                synapse_call_times [index] = clock.time() - start_time
                synapse_messages [index] = 'Input deserialization exception with error:{}'.format(str(e))
        # Check if the call can stop here.
        if self.check_if_should_return(synapse_codes=synapse_codes):
            self.finalize_codes_stats_and_logs()
            return [], synapse_codes[0] , request.synapses


        # ===================================
        # ==== Make forward calls. =========
        # ===================================
        try:
            self.finalize_codes_stats_and_logs()

            forward_response_tensors, forward_codes, forward_messages = self.module(
                inputs = deserialized_forward_tensors,
                synapses = synapses,
                hotkey= request.hotkey
            )

            synapse_is_response = [ True for _ in synapses ]
            # ========================================
            # ==== Fill codes from forward calls ====
            # ========================================
            for index, synapse in enumerate(synapses):
                synapse_codes [ index ] = forward_codes [ index ]
                synapse_messages [index] = forward_messages [ index ]
        # ========================================
        # ==== Catch forward request timeouts ====
        # ========================================
        except concurrent.futures.TimeoutError:
            code = bittensor.proto.ReturnCode.Timeout
            call_time = clock.time() - start_time
            message = "Request reached timeout"
            synapse_codes = [code for _ in synapses ]
            synapse_call_times = [call_time for _ in synapses ]
            synapse_messages = [ message for _ in synapses ]
            self.finalize_codes_stats_and_logs()
            return [], bittensor.proto.ReturnCode.Timeout, request.synapses

        # ==================================
        # ==== Catch unknown exceptions ====
        # ==================================
        except Exception as e:
            code = bittensor.proto.ReturnCode.UnknownException
            call_time = clock.time() - start_time
            synapse_codes = [code for _ in synapses ]
            synapse_call_times = [call_time for _ in synapses ]
            synapse_messages = [ 'Exception on Server' for _ in synapses ]
            self.finalize_codes_stats_and_logs(message = str(e))
            return [], bittensor.proto.ReturnCode.UnknownException, request.synapses

        # =================================================
        # ==== Encode/serialize responses and synapses ====
        # ==================================================
        response_synapses = []
        for index, synapse in enumerate( synapses ):
            try:
                if synapse_codes[index] == bittensor.proto.ReturnCode.Success:
                    synapse_responses [ index ] = synapse.serialize_forward_response_tensor( deserialized_forward_tensors[ index ], forward_response_tensors [ index ] )
                else:
                    synapse_responses [ index ] = synapse.empty()

            except ValueError as e:
                if str(e) == 'Empty Response':
                    synapse_codes [ index ]= bittensor.proto.ReturnCode.EmptyResponse
                else:
                    synapse_codes [ index ]= bittensor.proto.ReturnCode.ResponseShapeException

                synapse_call_times [ index ] = clock.time() - start_time
                synapse_messages [index] = "Synapse response shape exception with error: {}".format( str( e ) )
                synapse_responses [ index ] = synapse.empty()

            except Exception as e:
                synapse_codes [ index ]= bittensor.proto.ReturnCode.ResponseSerializationException
                synapse_call_times [ index ] = clock.time() - start_time
                synapse_messages [index] = "Synapse response serialization exception with error: {}".format( str( e ) )
                synapse_responses [ index ] = synapse.empty()

            response_synapses.append(synapse.serialize_to_wire_proto(code = synapse_codes[index], message= synapse_messages[index] ))

            
        # Check if the call can stop here.
        if self.check_if_should_return():
            self.finalize_codes_stats_and_logs()
            return [], synapse_codes[0], request.synapses

        # =========================================================
        # ==== Set return times for successfull forward ===========
        # =========================================================
        for index, _ in enumerate( synapses ):
            if synapse_codes[index] == bittensor.proto.ReturnCode.Success:
                synapse_call_times[index] = clock.time() - start_time

        self.finalize_codes_stats_and_logs()
        return synapse_responses, bittensor.proto.ReturnCode.Success, response_synapses
 
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

    def start(self) -> 'AxonGeneral':
        r""" Starts the standalone axon GRPC server thread.
        """
        if self.server != None:
            self.server.stop( grace = 1 )  
            logger.success("Axon Stopped:".ljust(20) + "<blue>{}</blue>", self.ip + ':' + str(self.port))

        self.server.start()
        logger.success("Axon Started:".ljust(20) + "<blue>{}</blue>", self.ip + ':' + str(self.port))
        self.started = True

        return self

    def stop(self) -> 'Axon':
        r""" Stop the axon grpc server.
        """
        if self.server != None:
            self.server.stop( grace = 1 )
            logger.success("Axon Stopped:".ljust(20) + "<blue>{}</blue>", self.ip + ':' + str(self.port))
        self.started = False

        return self
    
    def _init_stats(self):
        return SimpleNamespace(
            # Queries per second.
            qps = stat_utils.EventsPerSecondRollingAverage( 0, 0.01 ),
            # Total requests.
            total_requests = 0,
            # Total bytes recieved per second.
            total_in_bytes = 0,
            # Total bytes responded per second.
            total_out_bytes = 0,
            # Bytes recieved per second.
            avg_in_bytes_per_second = stat_utils.AmountPerSecondRollingAverage( 0, 0.01 ),
            # Bytes responded per second.
            avg_out_bytes_per_second = stat_utils.AmountPerSecondRollingAverage( 0, 0.01 ),
            # Requests per pubkey.
            requests_per_pubkey = {},
            # Success per pubkey.
            successes_per_pubkey = {},
            # Query time per pubkey.
            query_times_per_pubkey = {},
            # Queries per second per pubkey.
            qps_per_pubkey = {},
            # Codes recieved per pubkey.
            codes_per_pubkey = {},
            # Bytes recieved per pubkey.
            avg_in_bytes_per_pubkey = {},
            # Bytes sent per pubkey.
            avg_out_bytes_per_pubkey = {}
        )

    #TODO: Replace/update axon and dendrite stats 
    def update_stats_for_request(self, request, response, time, code):
        r""" Updates statistics for this request and response.
            Args:
                requests ( bittensor.proto.TensorMessage, `required`):
                    The request.
                response ( bittensor.proto.TensorMessage, `required`):
                    The response.
                time (:type:`float`, `required`):
                    Length of call in seconds.
                code (:obj:`bittensor.proto.ReturnCode, `required`)
                    Return code associated with the call i.e. Success of Timeout.
        """
        self.stats.qps.event()
        self.stats.total_requests += 1
        self.stats.total_in_bytes += sys.getsizeof(request) 
        self.stats.total_out_bytes += sys.getsizeof(response) 
        self.stats.avg_in_bytes_per_second.event( float(sys.getsizeof(request)) )
        self.stats.avg_out_bytes_per_second.event( float(sys.getsizeof(response)) )
        pubkey = request.hotkey
        if pubkey not in self.stats.requests_per_pubkey:
            self.stats.requests_per_pubkey[ pubkey ] = 0
            self.stats.successes_per_pubkey[ pubkey ] = 0
            self.stats.query_times_per_pubkey[ pubkey ] = stat_utils.AmountPerSecondRollingAverage(0, 0.05)
            self.stats.qps_per_pubkey[ pubkey ] = stat_utils.EventsPerSecondRollingAverage(0, 0.05)
            self.stats.codes_per_pubkey[ pubkey ] = dict([(k,0) for k in bittensor.proto.ReturnCode.keys()])
            self.stats.avg_in_bytes_per_pubkey[ pubkey ] = stat_utils.AmountPerSecondRollingAverage(0, 0.01)
            self.stats.avg_out_bytes_per_pubkey[ pubkey ] = stat_utils.AmountPerSecondRollingAverage(0, 0.01)

        # Add values.
        self.stats.requests_per_pubkey[ pubkey ] += 1
        self.stats.successes_per_pubkey[ pubkey ] += 1 if code == 1 else 0
        self.stats.query_times_per_pubkey[ pubkey ].event( float(time) )
        self.stats.avg_in_bytes_per_pubkey[ pubkey ].event( float(sys.getsizeof(request)) )
        self.stats.avg_out_bytes_per_pubkey[ pubkey ].event( float(sys.getsizeof(response)) )
        self.stats.qps_per_pubkey[ pubkey ].event()    
        try:
            if bittensor.proto.ReturnCode.Name( code ) in self.stats.codes_per_pubkey[ pubkey ].keys():
                self.stats.codes_per_pubkey[ pubkey ][bittensor.proto.ReturnCode.Name( code )] += 1
        except:
            pass  


    def to_dataframe ( self, metagraph ):
        r""" Return a stats info as a pandas dataframe indexed by the metagraph or pubkey if not existend.
            Args:
                metagraph: (bittensor.Metagraph):
                    Indexes the stats data using uids.
            Return:
                dataframe (:obj:`pandas.Dataframe`)
        """
        # Reindex the pubkey to uid if metagraph is present.
        try:
            index = [ metagraph.hotkeys.index(pubkey) for pubkey in self.stats.requests_per_pubkey.keys() if pubkey in metagraph.hotkeys ]
            columns = [ 'axon_n_requested', 'axon_n_success', 'axon_query_time','axon_avg_inbytes','axon_avg_outbytes', 'axon_qps' ]
            dataframe = pandas.DataFrame(columns = columns, index = index)
            for pubkey in self.stats.requests_per_pubkey.keys():
                if pubkey in metagraph.hotkeys:
                    uid = metagraph.hotkeys.index(pubkey)
                    dataframe.loc[ uid ] = pandas.Series( {
                        'axon_n_requested': int(self.stats.requests_per_pubkey[pubkey]),
                        'axon_n_success': int(self.stats.requests_per_pubkey[pubkey]),
                        'axon_query_time': float(self.stats.query_times_per_pubkey[pubkey].get()),             
                        'axon_avg_inbytes': float(self.stats.avg_in_bytes_per_pubkey[pubkey].get()),
                        'axon_avg_outbytes': float(self.stats.avg_out_bytes_per_pubkey[pubkey].get()),
                        'axon_qps': float(self.stats.qps_per_pubkey[pubkey].get())
                    } )
            dataframe['uid'] = dataframe.index
            return dataframe

        except Exception as e:
            bittensor.logging.error(prefix='failed axon.to_dataframe()', sufix=str(e))
            return pandas.DataFrame()
