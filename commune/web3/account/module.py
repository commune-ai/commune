#
# Copyright 2022 Ocean Protocol Foundation
# SPDX-License-Identifier: Apache-2.0
#
import logging
import os
from typing import Dict, Optional, Union
import json
from eth_account.datastructures import SignedMessage
from eth_account.messages import SignableMessage
from hexbytes.main import HexBytes
from web3.main import Web3
import streamlit as st
import gradio as gr
from commune import Module
from eth_account.messages import encode_defunct


from eth_keys import keys
from copy import deepcopy

def private_key_to_public_key(private_key: str) -> str:
    private_key_bytes = decode_hex(private_key)
    private_key_object = keys.PrivateKey(private_key_bytes)
    return private_key_object.public_key



logger = logging.getLogger(__name__)
from eth_account.account import Account

class AccountModule(Module):

    """
    The AccountModule is responsible for signing transactions and messages by using an self.account's
    private key.

    The use of this AccountModule allows Ocean tools to send rawTransactions which keeps the user
    key and password safe and they are never sent outside. Another advantage of this is that
    we can interact directly with remote network nodes without having to run a local parity
    node since we only send the raw transaction hash so the user info is safe.

    Usage:
    ```python
    AccountModule = AccountModule(
        ocean.web3,
        private_key=private_key,
        block_confirmations=ocean.config.block_confirmations,
        transaction_timeout=config.transaction_timeout,
    )
    ```

    """

    _last_tx_count = dict()
    ENV_PRIVATE_KEY = 'PRIVATE_KEY'
    def __init__(
        self,
        private_key: str= None,
        web3: Web3 = None,
        **kwargs
    ) -> None:
        """Initialises AccountModule object."""
        # assert private_key, "private_key is required."
        Module.__init__(self, **kwargs)


        self.account = self.set_account(private_key = private_key)
        self.web3 = web3

    @property
    def address(self) -> str:
        return self.account.address


    @property
    def private_key(self):
        return self.account._private_key
        
    def set_account(self, private_key=None):
        if isinstance(private_key, int):
            index = private_key
            private_key = list(self.accounts.keys())[i]
        elif isinstance(private_key, str):
            if isinstance(self.accounts, dict) \
                and private_key in self.accounts.keys():
                private_key = self.accounts[private_key]
            else:
                private_key = os.getenv(private_key, private_key) if isinstance(private_key, str) else None
                if private_key == None:
                    private_key = self.config.get('private_key', None)

        
        assert isinstance(private_key, str), f'private key should be string but is {type(private_key)}'


        self.account = Account.from_key(private_key)
        return self.account

    def set_web3(self, web3):
        self.web3 = web3
        return self.web3

    @property
    def key(self) -> str:
        return self.private_key

    @staticmethod
    def reset_tx_count() -> None:
        AccountModule._last_tx_count = dict()

    def validate(self, address:str) -> bool:
        return self.account.address == address

    @staticmethod
    def _get_nonce(web3: Web3, address: str) -> int:
        # We cannot rely on `web3.eth.get_transaction_count` because when sending multiple
        # transactions in a row without wait in between the network may not get the chance to
        # update the transaction count for the self.account address in time.
        # So we have to manage this internally per self.account address.
        if address not in AccountModule._last_tx_count:
            AccountModule._last_tx_count[address] = self.web3.eth.get_transaction_count(address)
        else:
            AccountModule._last_tx_count[address] += 1

        return AccountModule._last_tx_count[address]


    @property
    def address(self):
        return self.account.address

    def sign_tx(
        self,
        tx: Dict[str, Union[int, str, bytes]],
    ) -> HexBytes:
        if tx.get('nonce') == None:
            nonce = AccountModule._get_nonce(self.web3, self.address)
        if tx.get('gasePrice') == None:
            gas_price = int(self.web3.eth.gas_price * 1.1)
            max_gas_price = os.getenv('ENV_MAX_GAS_PRICE', None)
            if gas_price and max_gas_price:
                gas_price = min(gas_price, max_gas_price)

            tx["gasPrice"] = gas_price


        signed_tx = self.web3.eth.account.sign_transaction(tx, self.private_key)
        logger.debug(f"Using gasPrice: {gas_price}")
        logger.debug(f"`AccountModule` signed tx is {signed_tx}")
        return signed_tx.rawTransaction

    @property
    def nonce(self):
        return self.web3.eth.get_transaction_count(self.address)

    @property
    def tx_metadata(self):
        return {
        'from': self.address,
        'nonce': self.nonce,
        'gasPrice':self.web3.eth.generate_gas_price(),
        }
    def send_contract_tx(self, fn , value=0):
        tx_metadata = self.tx_metadata
        tx_metadata['value'] = value

        tx = fn.buildTransaction(
            tx_metadata
        )

        tx =  self.send_tx(tx)
        return tx
    def send_tx(self, tx):
        
        rawTransaction = self.sign_tx(tx=tx)        
        # 7. Send tx and wait for receipt
        tx_hash = self.web3.eth.send_raw_transaction(rawTransaction)
        tx_receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)

        return tx_receipt.__dict__

    @staticmethod
    def python2str(input):
        input = deepcopy(input)
        input_type = type(input)
        if input_type in [dict]:
            input = json.dumps(input)
        elif input_type in [list, tuple, set]:
            input = json.dumps(list(input))
        elif message_type in [int, float, bool]:
            input = str(input)
        return message

    @staticmethod
    def str2python(input)-> dict:
        assert isinstance(input, str)
        output_dict = json.loads(input)
        return output_dict
    
    def resolve_message(self, message):
        message = self.python2str(message)


        if isinstance(msg_hash, str):
            message = encode_defunct(message)
        elif isinstance(message, SignableMessage):
            message = message
        else:
            raise NotImplemented
            

    def sign(self, message: Union[SignableMessage,str, dict]) -> SignedMessage:
        """Sign a transaction."""
        message = self.resolve_message(message)
        return self.account.sign_message(message)

    @property
    def public_key(self):
        return private_key_to_public_key(self.private_key)
        
    def keys_str(self) -> str:
        s = []
        s += [f"address: {self.address}"]
        if self.private_key is not None:
            s += [f"private key: {self.private_key}"]
            s += [f"public key: {self.public_key}"]
        s += [""]
        return "\n".join(s)


    hash_fn_dict = {
        'keccak': Web3.keccak
    }
    @staticmethod
    def resolve_hash_function(cls, hash_type='keccak'):
        hash_fn = AccountModule.hash_fn_dict.get(hash_type)
        assert hash_fn != None, f'hash_fn: {hash_type} is not found'
        return hash_fn

    @staticmethod
    def hash(input, hash_type='keccak',return_type='str',*args,**kwargs):
        
        hash_fn = AccountModule.resolve_hash_function(hash_type)

        input = AccountModule.python2str(input)
        hash_output = Web3.keccak(text=input, *args, **kwargs)
        if return_type in ['str', str, 'string']:
            hash_output = Web3.toHex(hash_output)
        elif return_type in ['hex', 'hexbytes']:
            pass
        else:
            raise NotImplementedError(return_type)
        
        return hash_output

    
    def resolve_web3(self, web3=None):
        if web3 == None:
            web3 == self.web3
        assert web3 != None
        return web3

    def resolve_address(self, address=None):
        if address == None:
            address == self.address
        assert address != None
        return address


    def get_balance(self, token:str=None, address=None, web3=None):
        web3 = self.resolve_web3(web3)
        address = self.resolve_address(address)
        
        if token == None:
            # return native token
            balance = self.web3.eth.get_balance(self.address)
        else:
            raise NotImplemented

        return balance

    @property
    def accounts(self):
        return self.config.get('accounts', [])
        

    @classmethod
    def streamlit(cls):
        st.write(f'### {cls.__name__}')
        self = cls.deploy(actor={'refresh': False, 'wrap': True})


    def replicate(self, private_key, web3=None):
        return AccountModule(private_key=private_key, web3=self.web3)
        




if __name__ == '__main__':
    AccountModule.streamlit()
    # module.gradio()


