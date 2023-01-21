# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

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

import bittensor
import torch
from typing import Union, List, Tuple, Optional
from .synapse_impl import Synapse


class GeneralJSON (Synapse):
    """  Synapse type for next token prediction from languge models.
    """
    synapse_type: bittensor.proto.Synapse.SynapseType = bittensor.proto.Synapse.SynapseType.GENERAL_JSON

    def __init__( 
        self, 
        forward_request_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        forward_response_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        backward_request_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        backward_response_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
    ):  
        """ TextCausalLM Synapse initializer.
        Args:
            forward_request_serializer_type (:obj:`bittensor.proto.Serializer.Type` of shape :obj:`(1)`, `optional`, :default: `bittensor.proto.Serializer.MSGPACK`):
                Serializer used to pack torch tensors on forward request.
            forward_response_serializer_type (:obj:`bittensor.proto.Serializer.Type` of shape :obj:`(1)`, `optional`, :default: `bittensor.proto.Serializer.MSGPACK`):
                Serializer used to pack torch tensors on forward response.
            backward_request_serializer_type (:obj:`bittensor.proto.Serializer.Type` of shape :obj:`(1)`, `optional`, :default: `bittensor.proto.Serializer.MSGPACK`):
                Serializer used to pack torch tensors on forward request.
            backward_response_serializer_type (:obj:`bittensor.proto.Serializer.Type` of shape :obj:`(1)`, `optional`, :default: `bittensor.proto.Serializer.MSGPACK`):
                Serialzer used to pack torch tensors on backward response.
        Returns:
            TextLastHiddenState (:obj:`TextLastHiddenState`, `required`):
                TextLastHiddenState instance adapter class.
    """
        super().__init__ (
            forward_request_serializer_type,
            forward_response_serializer_type,
            backward_request_serializer_type,
            backward_response_serializer_type
        )
        self.synapse_type = GeneralJSON.synapse_type

    def __repr__(self) -> str: return self.__str__()
    def __str__(self) -> str: return "TextCausalLM"

    @staticmethod
    def deserialize_from_instance_proto ( instance_proto: bittensor.proto.Synapse ) -> 'GeneralJSON':
        return GeneralJSON ( 
            forward_request_serializer_type = instance_proto.forward_request_serializer_type,
            forward_response_serializer_type = instance_proto.forward_response_serializer_type,
            backward_request_serializer_type = instance_proto.backward_request_serializer_type,
            backward_response_serializer_type = instance_proto.backward_response_serializer_type,
        )

    @staticmethod
    def deserialize_from_wire_proto ( wire_proto: bittensor.proto.Synapse ) -> 'GeneralJSON':
        instance_proto = bittensor.proto.Synapse.GeneralJSON()
        instance_proto.ParseFromString( wire_proto.synapse_data )
        return GeneralJSON.deserialize_from_instance_proto( instance_proto )

    def serialize_to_instance_proto( self ) -> 'bittensor.proto.Synapse.GeneralJSON':
        return bittensor.proto.Synapse.GeneralJSON ( 
            forward_request_serializer_type = self.forward_request_serializer_type,
            forward_response_serializer_type = self.forward_response_serializer_type,
            backward_request_serializer_type = self.backward_request_serializer_type,
            backward_response_serializer_type = self.backward_response_serializer_type,
        )
        
    def serialize_to_wire_proto ( self, code: 'bittensor.proto.ReturnCode' = 0, message: str = '' ) -> bittensor.proto.Synapse:
        return bittensor.proto.Synapse (
                synapse_data = self.serialize_to_instance_proto().SerializeToString(),
                synapse_type = GeneralJSON.synapse_type,
                return_code = code,
                message = message
            )

    def check_forward_request_tensor ( self, forward_request_tensor ): 
        assert isinstance(forward_request_tensor, dict)

    def check_forward_response_tensor ( self, forward_request_tensor, forward_response_tensor ):
        assert isinstance(forward_request_tensor, dict)
        assert isinstance(forward_response_tensor, dict)

    def check_backward_request_gradient ( self, forward_request_tensor, backward_request_gradient ):
        assert isinstance(forward_request_tensor, dict)
        assert isinstance(forward_response_tensor, dict)   

    def encode_forward_request_tensor ( self, forward_request_tensor: dict) -> dict: return forward_request_tensor
    def decode_forward_request_tensor ( self, forward_request_tensor: dict ) -> dict: return forward_request_tensor
    def encode_forward_response_tensor( self, forward_response_tensor: dict ) -> dict: return forward_response_tensor
    def decode_forward_response_tensor( self, forward_response_tensor: dict ) -> dict: return forward_response_tensor
    def encode_backward_response_gradient( self, backward_request_gradient: dict ) -> dict: return backward_request_gradient
    def decode_backward_response_gradient ( self, backward_request_gradient: dict ) -> dict: return backward_request_gradient
    def encode_backward_request_gradient( self, backward_response_gradient: dict ) -> dict: return backward_response_gradient
    def decode_backward_request_gradient( self, backward_response_gradient: dict ) -> dict: return backward_response_gradient
    def nill_forward_response_tensor( self, forward_request_tensor: dict ) -> dict: return {}
    def nill_backward_response_tensor( self, forward_request_tensor: dict ) -> dict: return {}

    def serialize_forward_request_tensor( self, forward_request_tensor: torch.Tensor ) -> Tuple[ 'bittensor.proto.Tensor', 'bittensor.proto.ReturnCode',  str ]:        
        self.check_forward_request_tensor ( forward_request_tensor )
        forward_request_tensor = self.encode_forward_request_tensor ( forward_request_tensor )
        tensor_serialzier = bittensor.serializer( serializer_type = self.forward_request_serializer_type )
        return tensor_serialzier.serialize( tensor_obj = forward_request_tensor, from_type = bittensor.proto.TensorType.JSON )

    def deserialize_forward_request_tensor( self, forward_request_proto: bittensor.proto.Tensor ) -> Tuple[ 'torch.Tensor', 'bittensor.proto.ReturnCode',  str ]:
        """ Returns a torch.Tensor from wire proto.Tensor after relevant deserialization has been applied. """
        tensor_deserialzier = bittensor.serializer( serializer_type = self.forward_request_serializer_type )
        forward_request_tensor = tensor_deserialzier.deserialize( tensor_pb2 = forward_request_proto, to_type = bittensor.proto.TensorType.JSON )
        forward_request_tensor = self.decode_forward_request_tensor ( forward_request_tensor )
        self.check_forward_request_tensor ( forward_request_tensor )
        return forward_request_tensor

    def serialize_forward_response_tensor( self, forward_request_tensor: torch.Tensor, forward_response_tensor: dict ) -> Tuple[ 'bittensor.proto.Tensor', 'bittensor.proto.ReturnCode',  str ]:
        """ Returns a bittensor.proto.Tensor to be sent on the wire after relevant serialization applied. """  
        encoded_tensor = self.encode_forward_response_tensor ( forward_response_tensor )
        self.check_forward_response_tensor ( forward_request_tensor, encoded_tensor )
        tensor_serialzier = bittensor.serializer( serializer_type = self.forward_response_serializer_type )
        return tensor_serialzier.serialize( tensor_obj = encoded_tensor, from_type = bittensor.proto.TensorType.JSON )
    
    def deserialize_forward_response_proto( self, forward_request_tensor: dict, forward_response_proto: dict ) -> Tuple[ dict, 'bittensor.proto.ReturnCode',  str ]:
        """ Returns a torch.Tensor from wire proto.Tensor after relevant deserialization has been applied. """
        tensor_deserialzier = bittensor.serializer( serializer_type = self.forward_response_serializer_type )
        forward_response_tensor = tensor_deserialzier.deserialize( tensor_pb2 = forward_response_proto, to_type = bittensor.proto.TensorType.JSON )
        self.check_forward_response_tensor ( forward_request_tensor, forward_response_tensor )
        forward_response_tensor = self.decode_forward_response_tensor ( forward_response_tensor )
        return forward_response_tensor

    def serialize_backward_request_gradient( self, forward_request_tensor: dict, backward_request_gradient: dict ) -> Tuple[ 'bittensor.proto.Tensor', 'bittensor.proto.ReturnCode',  str ]:
        """ Returns a bittensor.proto.Tensor gradient to be sent on the wire after relevant serialization applied. """
        self.check_backward_request_gradient ( forward_request_tensor, backward_request_gradient )
        encoded_tensor = self.encode_backward_request_gradient ( backward_request_gradient )
        tensor_serialzier = bittensor.serializer( serializer_type = self.forward_request_serializer_type )
        return tensor_serialzier.serialize( tensor_obj = encoded_tensor, from_type = bittensor.proto.TensorType.JSON )

    def deserialize_backward_request_gradient( self, forward_request_tensor: torch.Tensor, backward_request_proto: bittensor.proto.Tensor ) -> Tuple[ 'torch.Tensor', 'bittensor.proto.ReturnCode',  str ]:
        tensor_deserialzier = bittensor.serializer( serializer_type = self.backward_request_serializer_type )
        backward_request_gradient = tensor_deserialzier.deserialize( tensor_pb2 = backward_request_proto, to_type = bittensor.proto.TensorType.JSON )
        backward_request_gradient = self.decode_backward_request_gradient ( backward_request_gradient )
        self.check_backward_request_gradient (forward_request_tensor,  backward_request_gradient )
        return backward_request_gradient
