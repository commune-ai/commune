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
from eth_account.messages import encode_defunct
from eth_keys import keys
from copy import deepcopy
from eth_account import Account

import commune as c
from typing import List, Dict, Union, Optional, Any

logger = logging.getLogger(__name__)


class EVMAccount(c.Module, Account):

    _last_tx_count = dict()
    def __init__(
        self,
        network:str = 'local.main',
        **kwargs
    ) -> None:
        """Initialises EVMAccount object."""
        # assert private_key, "private_key is required."
        self.config = self.set_config( kwargs=kwargs)
        Account.__init__(self, **kwargs)
        self.set_network(network)


    @property
    def private_key(self):
        return self._private_key

    @staticmethod
    def reset_tx_count() -> None:
        EVMAccount._last_tx_count = dict()

    def get_nonce(self, address: str = None) -> int:
        # We cannot rely on `web3.eth.get_transaction_count` because when sending multiple
        # transactions in a row without wait in between the network may not get the chance to
        # update the transaction count for the self address in time.
        # So we have to manage this internally per self address.
        address = self.resolve_address(address)
        if address not in EVMAccount._last_tx_count:
            EVMAccount._last_tx_count[address] = self.web3.eth.get_transaction_count(address)
        else:
            EVMAccount._last_tx_count[address] += 1

        return EVMAccount._last_tx_count[address]


    def sign_tx(
        self,
        tx: Dict[str, Union[int, str, bytes]],
    ) -> HexBytes:
        if tx.get('nonce') == None:
            nonce = self.get_nonce(web3=self.web3, address=self.address)
        if tx.get('gasePrice') == None:
            gas_price = int(self.web3.eth.gas_price * 1.1)
            max_gas_price = os.getenv('ENV_MAX_GAS_PRICE', None)
            if gas_price and max_gas_price:
                gas_price = min(gas_price, max_gas_price)

            tx["gasPrice"] = gas_price


        signed_tx = self.web3.eth.account.sign_transaction(tx, self.private_key)
        logger.debug(f"Using gasPrice: {gas_price}")
        logger.debug(f"`EVMAccount` signed tx is {signed_tx}")
        return signed_tx.rawTransaction

    @property
    def nonce(self):
        return self.web3.eth.get_transaction_count(self.address)

    @property
    def gas_prices(self):
        return self.web3.eth.generate_gas_price()

    @property
    def tx_metadata(self) -> Dict[str, Union[int, str, bytes]]:
        '''
        Default tx metadata
        '''
        
        return {
        'from': self.address,
        'nonce': self.nonce,
        'gasPrice':self.gas_price,
        }
    def send_contract_tx(self, fn:str , value=0):
        '''
        send a contract transaction for your python objecs
        '''
        tx_metadata = self.tx_metadata
        tx_metadata['value'] = value
        tx = fn.buildTransaction(tx_metadata)
        tx =  self.send_tx(tx)
        return tx
    
    def send_tx(self, tx):
        '''
        Send a transaction
        '''
        rawTransaction = self.sign_tx(tx=tx)        
        # 7. Send tx and wait for receipt
        tx_hash = self.web3.eth.send_raw_transaction(rawTransaction)
        tx_receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)

        return tx_receipt.__dict__

    
    def resolve_message(self, message) :
        message = self.python2str(message)
        if isinstance(message, str):
            message = encode_defunct(text=message)
        elif isinstance(message, SignableMessage):
            message = message
        else:
            raise NotImplemented
        
        return message
            

    def sign(self, message: Union[SignableMessage,str, dict], include_message:bool = True) -> SignedMessage:
        """Sign a transaction.
        Args:
            message: The message to sign.
            signature_only: If True, only the signature is returned.
        """
        signable_message = self.resolve_message(message)

        signed_message = self.sign_message(signable_message)
        signed_message_dict = {}
        for k in ['v', 'r', 's', 'signature', 'messageHash']:
            signed_message_dict[k] = getattr(signed_message, k)
            if isinstance(signed_message_dict[k], HexBytes):
                signed_message_dict[k] = signed_message_dict[k].hex()
                
        if include_message:
            signed_message_dict['message'] = message
        signed_message = signed_message_dict
        
        
        return signed_message

    @property
    def public_key(self):
        return self.private_key_to_public_key(self.private_key)
    
    
    @staticmethod
    def private_key_to_public_key(private_key: str) -> str:
        '''
        Conert private key to public key
        '''
        private_key_object = keys.PrivateKey(private_key)
        return private_key_object.public_key


  
    def keys_str(self) -> str:
        s = []
        s += [f"address: {self.address}"]
        if self.private_key is not None:
            s += [f"private key: {self.private_key}"]
            s += [f"public key: {self.public_key}"]
        s += [""]
        return "\n".join(s)

    def resolve_web3(self, web3=None):
        if web3 == None:
            web3 == self.web3
        assert web3 != None
        return web3

    def resolve_address(self, address=None):
        if address == None:
            address =  self.address
        assert address != None
        return address


    def get_balance(self, token:str=None, address:str=None):
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
        import streamlit as st
        st.write(f'### {cls.__name__}')
        self = cls.deploy(actor={'refresh': False, 'wrap': True})


    def replicate(self, private_key, web3=None):
        return EVMAccount(private_key=private_key, web3=self.web3)
        

    def set_network(self, network:str= 'local.main') -> None:
        '''
        Set network
        '''
        if isinstance(network, str):
            network = {
                'module': 'web3.evm.network',
                'kwargs': {
                    'network': network
                } 
            }
        if network == None:
            network = self.config['network']
            
        # launch network
        self.network = self.launch(**network)
        self.web3 = self.network.web3

    @staticmethod
    def hex2str(input:HexBytes) -> str:
        '''
        HexBytes to str
        '''
        return input.hex()

    def recover_signer(self, message:Any, 
                        signature:str, 
                        vrs:Union[tuple, list]=None):
        '''
        recover
        '''
        
        message = self.resolve_message(message)
        recovered_address = Account.recover_message(message, signature=signature, vrs=vrs)
        return recovered_address
    
    def verify(self, message:Any, signature:str = None, vrs:Union[tuple, list]=None, address:str=None) -> bool:
        '''
        verify message from the signature or vrs based on the address
        '''
        address = self.resolve_address(address)
        recovered_address = self.recover_signer(message, signature=signature, vrs=vrs)
        return bool(recovered_address == address)

    @classmethod
    def test_sign(cls):
        self = cls()
        message = {'bro': 'bro'}
        signature = self.sign(message)
        assert self.verify(message, signature=signature['signature'])
        
    @classmethod
    def test_hash(cls):
        self = cls()
        print(self.hash('hello world'))
        
        
    def test(self):
        self.test_sign()
        self.test_hash()
        self.test_recover_message()
        self.test_verify_message()
        
       
    @classmethod
    def from_password(cls, password:str, salt:str='commune', prompt=False):
        
        from web3.auto import w3
        from Crypto.Protocol.KDF import PBKDF2

        # Prompt the user for a password and salt
        if prompt :
            password = input("Enter password: ")
        # Derive a key using PBKDF2
        key = PBKDF2(password.encode(), salt, dkLen=32, count=100000)

        # Create an account using the key
        account = Account.privateKeyToAccount(key)

        # Print the account address and private key
        print("Account address:", account.address)
        print("Private key:", account.privateKey.hex())
        
        return account

if __name__ == '__main__':
    EVMAccount.run()



