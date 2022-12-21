import argparse
import os
import copy
import inspect
import time
from concurrent import futures
from typing import Dict, List, Callable, Optional, Tuple, Union

import torch
import grpc
from substrateinterface import Keypair

import bittensor



class ServerInterceptor(grpc.ServerInterceptor):
    """Creates a new server interceptor that authenticates incoming messages from passed arguments."""

    def __init__(
        self,
        receiver_hotkey: str,
        blacklist: Callable = None,
    ):
        r"""Creates a new server interceptor that authenticates incoming messages from passed arguments.
        Args:
            receiver_hotkey(str):
                the SS58 address of the hotkey which should be targeted by RPCs
            black_list (Function, `optional`):
                black list function that prevents certain pubkeys from sending messages
        """
        super().__init__()
        self.nonces = {}
        self.blacklist = blacklist
        self.receiver_hotkey = receiver_hotkey

    def parse_legacy_signature(
        self, signature: str
    ) -> Union[Tuple[int, str, str, str, int], None]:
        r"""Attempts to parse a signature using the legacy format, using `bitxx` as a separator"""
        parts = signature.split("bitxx")
        if len(parts) < 4:
            return None
        try:
            nonce = int(parts[0])
            parts = parts[1:]
        except ValueError:
            return None
        receptor_uuid, parts = parts[-1], parts[:-1]
        signature, parts = parts[-1], parts[:-1]
        sender_hotkey = "".join(parts)
        return (nonce, sender_hotkey, signature, receptor_uuid, 1)

    def parse_signature_v2(
        self, signature: str
    ) -> Union[Tuple[int, str, str, str, int], None]:
        r"""Attempts to parse a signature using the v2 format"""
        parts = signature.split(".")
        if len(parts) != 4:
            return None
        try:
            nonce = int(parts[0])
        except ValueError:
            return None
        sender_hotkey = parts[1]
        signature = parts[2]
        receptor_uuid = parts[3]
        return (nonce, sender_hotkey, signature, receptor_uuid, 2)

    def parse_signature(
        self, metadata: Dict[str, str]
    ) -> Tuple[int, str, str, str, int]:
        r"""Attempts to parse a signature from the metadata"""
        signature = metadata.get("bittensor-signature")
        if signature is None:
            raise Exception("Request signature missing")
        for parser in [self.parse_signature_v2, self.parse_legacy_signature]:
            parts = parser(signature)
            if parts is not None:
                return parts
        raise Exception("Unknown signature format")

    def check_signature(
        self,
        nonce: int,
        sender_hotkey: str,
        signature: str,
        receptor_uuid: str,
        format: int,
    ):
        r"""verification of signature in metadata. Uses the pubkey and nonce"""
        keypair = Keypair(ss58_address=sender_hotkey)
        # Build the expected message which was used to build the signature.
        if format == 2:
            message = f"{nonce}.{sender_hotkey}.{self.receiver_hotkey}.{receptor_uuid}"
        elif format == 1:
            message = f"{nonce}{sender_hotkey}{receptor_uuid}"
        else:
            raise Exception("Invalid signature version")
        # Build the key which uniquely identifies the endpoint that has signed
        # the message.
        endpoint_key = f"{sender_hotkey}:{receptor_uuid}"

        if endpoint_key in self.nonces.keys():
            previous_nonce = self.nonces[endpoint_key]
            # Nonces must be strictly monotonic over time.
            if nonce <= previous_nonce:
                raise Exception("Nonce is too small")

        if not keypair.verify(message, signature):
            raise Exception("Signature mismatch")
        self.nonces[endpoint_key] = nonce

    def black_list_checking(self, hotkey: str, method: str):
        r"""Tries to call to blacklist function in the miner and checks if it should blacklist the pubkey"""
        if self.blacklist == None:
            return

        request_type = {
            "/Bittensor/Forward": bittensor.proto.RequestType.FORWARD,
            "/Bittensor/Backward": bittensor.proto.RequestType.BACKWARD,
        }.get(method)
        if request_type is None:
            raise Exception("Unknown request type")

        if self.blacklist(hotkey, request_type):
            raise Exception("Request type is blacklisted")

    def intercept_service(self, continuation, handler_call_details):
        r"""Authentication between bittensor nodes. Intercepts messages and checks them"""
        method = handler_call_details.method
        metadata = dict(handler_call_details.invocation_metadata)

        try:
            (
                nonce,
                sender_hotkey,
                signature,
                receptor_uuid,
                signature_format,
            ) = self.parse_signature(metadata)

            # signature checking
            self.check_signature(
                nonce, sender_hotkey, signature, receptor_uuid, signature_format
            )

            # blacklist checking
            self.black_list_checking(sender_hotkey, method)

            return continuation(handler_call_details)

        except Exception as e:
            message = str(e)
            abort = lambda _, ctx: ctx.abort(grpc.StatusCode.UNAUTHENTICATED, message)
            return grpc.unary_unary_rpc_method_handler(abort)
