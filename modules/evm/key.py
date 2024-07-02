#
# Copyright 2022 Ocean Protocol Foundation
# SPDX-License-Identifier: Apache-2.0
#
import os
from typing import *
from eth_account.datastructures import SignedMessage
from eth_account.messages import SignableMessage
from hexbytes.main import HexBytes
from eth_keys import keys
from eth_account import Account

import commune as c

from collections.abc import (
    Mapping,
)
import json
import os
from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)
import warnings

from cytoolz import (
    dissoc,
)
from eth_keyfile import (
    create_keyfile_json,
    decode_keyfile_json,
)
from eth_keys import (
    KeyAPI,
    keys,
)
from eth_keys.exceptions import (
    ValidationError,
)
from eth_typing import (
    ChecksumAddress,
    Hash32,
    HexStr,
)
from eth_utils.curried import (
    combomethod,
    hexstr_if_str,
    is_dict,
    keccak,
    text_if_str,
    to_bytes,
    to_int,
)
from hexbytes import (
    HexBytes,
)

from eth_account._utils.legacy_transactions import (
    Transaction,
    vrs_from,
)
from eth_account._utils.signing import (
    hash_of_signed_transaction,
    sign_message_hash,
    sign_transaction_dict,
    to_standard_signature_bytes,
    to_standard_v,
)
from eth_account._utils.typed_transactions import (
    TypedTransaction,
)
from eth_account.datastructures import (
    SignedMessage,
    SignedTransaction,
)
from eth_account.hdaccount import (
    ETHEREUM_DEFAULT_PATH,
    generate_mnemonic,
    key_from_seed,
    seed_from_mnemonic,
)
from eth_account.messages import (
    SignableMessage,
    _hash_eip191_message,
    encode_typed_data,
)
from eth_account.signers.local import (
    LocalAccount,
)

VRS = TypeVar("VRS", bytes, HexStr, int)

class EVMKey(c.Module):

    _last_tx_count = {}
    def __init__(
        self,
        network:str = 'local.main',
        **kwargs
    ) -> None:
        """Initialises EVMAccount object."""
        # assert private_key, "private_key is required."
        self.set_config( kwargs=locals())
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
            tx['nonce'] = self.nonce
        if tx.get('gasePrice') == None:
            gas_price = self.gas_price
            max_gas_price = os.getenv('ENV_MAX_GAS_PRICE', None)
            if gas_price and max_gas_price:
                gas_price = min(gas_price, max_gas_price)

            tx["gasPrice"] = gas_price
        signed_tx = self.web3.eth.account.sign_transaction(tx, self.private_key)
        return signed_tx.rawTransaction

    @property
    def nonce(self):
        return self.web3.eth.get_transaction_count(self.address)

    @property
    def gas_price(self):
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
        from eth_account.messages import encode_defunct
        message = c.python2str(message)
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
        

    def set_network(self, network:str= 'local.main') -> None:
        '''
        Set network
        '''
        self.web3 = c.module('web3.evm.network')(network=network).web3



    def recover_signer(self, message:Any, 
                        signature:str, 
                        vrs:Union[tuple, list]=None):
        '''
        recover
        '''
        
        message = self.resolve_message(message)
        recovered_address = self.recover_message(message, signature=signature, vrs=vrs)
        return recovered_address
    
    def verify(self, message:Any, signature:str = None, vrs:Union[tuple, list]=None, address:str=None) -> bool:
        '''
        verify message from the signature or vrs based on the address
        '''
        address = self.resolve_address(address)
        recovered_address = self.recover_signer(message, signature=signature, vrs=vrs)
        return bool(recovered_address == address)

       
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


    @classmethod
    def test_sign(cls):
        self = cls()
        message = {'bro': 'bro'}
        signature = self.sign(message)
        assert self.verify(message, signature=signature['signature'])
        
        
    def test(self):
        self.test_sign()
        

    _keys = keys

    _default_kdf = os.getenv("ETH_ACCOUNT_KDF", "scrypt")

    # Enable unaudited features (off by default)
    _use_unaudited_hdwallet_features = False

    @classmethod
    def enable_unaudited_hdwallet_features(cls):
        """
        Use this flag to enable unaudited HD Wallet features.
        """
        cls._use_unaudited_hdwallet_features = True

    @combomethod
    def create(self, extra_entropy=""):
        r"""
        Creates a new private key, and returns it as a
        :class:`~eth_account.local.LocalAccount`.

        :param extra_entropy: Add extra randomness to whatever randomness your OS
          can provide
        :type extra_entropy: str or bytes or int
        :returns: an object with private key and convenience methods

        """
        extra_key_bytes = text_if_str(to_bytes, extra_entropy)
        key_bytes = keccak(os.urandom(32) + extra_key_bytes)
        return self.from_key(key_bytes)

    @combomethod
    def from_key(self, private_key):
        r"""
        Returns a convenient object for working with the given private key.

        :param private_key: The raw private key
        :type private_key: hex str, bytes, int or :class:`eth_keys.datatypes.PrivateKey`
        :return: object with methods for signing and encrypting
        :rtype: LocalAccount

        .. doctest:: python

            >>> acct = Account.from_key(
            ... 0xb25c7db31feed9122727bf0939dc769a96564b2de4c4726d035b36ecf1e5b364)
            >>> acct.address
            '0x5ce9454909639D2D17A3F753ce7d93fa0b9aB12E'
            >>> acct.key
            HexBytes('0xb25c7db31feed9122727bf0939dc769a96564b2de4c4726d035b36ecf1e5b364')

            # These methods are also available: sign_message(), sign_transaction(),
            # encrypt(). They correspond to the same-named methods in Account.*
            # but without the private key argument
        """
        key = self._parsePrivateKey(private_key)
        return LocalAccount(key, self)

    @combomethod
    def from_mnemonic(
        self,
        mnemonic: str,
        passphrase: str = "",
        account_path: str = ETHEREUM_DEFAULT_PATH,
    ) -> LocalAccount:
        """
        Generate an account from a mnemonic.

        .. CAUTION:: This feature is experimental, unaudited, and likely to change soon

        :param str mnemonic: space-separated list of BIP39 mnemonic seed words
        :param str passphrase: Optional passphrase used to encrypt the mnemonic
        :param str account_path: Specify an alternate HD path for deriving the seed
            using BIP32 HD wallet key derivation.
        :return: object with methods for signing and encrypting
        :rtype: LocalAccount

        """
        if not self._use_unaudited_hdwallet_features:
            raise AttributeError(
                "The use of the Mnemonic features of Account is disabled by "
                "default until its API stabilizes. To use these features, please "
                "enable them by running `Account.enable_unaudited_hdwallet_features()` "
                "and try again."
            )
        seed = seed_from_mnemonic(mnemonic, passphrase)
        private_key = key_from_seed(seed, account_path)
        key = self._parsePrivateKey(private_key)
        return LocalAccount(key, self)

    @combomethod
    def create_with_mnemonic(
        self,
        passphrase: str = "",
        num_words: int = 12,
        language: str = "english",
        account_path: str = ETHEREUM_DEFAULT_PATH,
    ) -> Tuple[LocalAccount, str]:
        r"""
        Create a new private key and related mnemonic.

        .. CAUTION:: This feature is experimental, unaudited, and likely to change soon

        Creates a new private key, and returns it as a
        :class:`~eth_account.local.LocalAccount`, alongside the mnemonic that can
        used to regenerate it using any BIP39-compatible wallet.

        :param str passphrase: Extra passphrase to encrypt the seed phrase
        :param int num_words: Number of words to use with seed phrase.
                              Default is 12 words.
                              Must be one of [12, 15, 18, 21, 24].
        :param str language: Language to use for BIP39 mnemonic seed phrase.
        :param str account_path: Specify an alternate HD path for deriving the
            seed using BIP32 HD wallet key derivation.
        :returns: A tuple consisting of an object with private key and
                  convenience methods, and the mnemonic seed phrase that can be
                  used to restore the account.
        :rtype: (LocalAccount, str)

        .. doctest:: python

            >>> from eth_account import Account
            >>> Account.enable_unaudited_hdwallet_features()
            >>> acct, mnemonic = Account.create_with_mnemonic()
            >>> acct.address # doctest: +SKIP
            '0x5ce9454909639D2D17A3F753ce7d93fa0b9aB12E'
            >>> acct == Account.from_mnemonic(mnemonic)
            True

            # These methods are also available:
            # sign_message(), sign_transaction(), encrypt()
            # They correspond to the same-named methods in Account.*
            # but without the private key argument
        """
        if not self._use_unaudited_hdwallet_features:
            raise AttributeError(
                "The use of the Mnemonic features of Account is disabled by "
                "default until its API stabilizes. To use these features, please "
                "enable them by running `Account.enable_unaudited_hdwallet_features()` "
                "and try again."
            )
        mnemonic = generate_mnemonic(num_words, language)
        return self.from_mnemonic(mnemonic, passphrase, account_path), mnemonic

    @combomethod
    def recover_message(
        self,
        signable_message: SignableMessage,
        vrs: Optional[Tuple[VRS, VRS, VRS]] = None,
        signature: bytes = None,
    ) -> ChecksumAddress:
        r"""
        Get the address of the account that signed the given message.
        You must specify exactly one of: vrs or signature

        :param signable_message: the message that was signed
        :param vrs: the three pieces generated by an elliptic curve signature
        :type vrs: tuple(v, r, s), each element is hex str, bytes or int
        :param signature: signature bytes concatenated as r+s+v
        :type signature: hex str or bytes or int
        :returns: address of signer, hex-encoded & checksummed
        :rtype: str

        .. doctest:: python

            >>> from eth_account.messages import encode_defunct
            >>> from eth_account import Account
            >>> message = encode_defunct(text="I♥SF")
            >>> vrs = (
            ...   28,
            ...   '0xe6ca9bba58c88611fad66a6ce8f996908195593807c4b38bd528d2cff09d4eb3',
            ...   '0x3e5bfbbf4d3e39b1a2fd816a7680c19ebebaf3a141b239934ad43cb33fcec8ce')
            >>> Account.recover_message(message, vrs=vrs)
            '0x5ce9454909639D2D17A3F753ce7d93fa0b9aB12E'


            # All of these recover calls are equivalent:

            # variations on vrs
            >>> vrs = (
            ...   '0x1c',
            ...   '0xe6ca9bba58c88611fad66a6ce8f996908195593807c4b38bd528d2cff09d4eb3',
            ...   '0x3e5bfbbf4d3e39b1a2fd816a7680c19ebebaf3a141b239934ad43cb33fcec8ce')
            >>> Account.recover_message(message, vrs=vrs)
            '0x5ce9454909639D2D17A3F753ce7d93fa0b9aB12E'

            >>> # Caution about this approach: likely problems if there are leading 0s
            >>> vrs = (
            ...   0x1c,
            ...   0xe6ca9bba58c88611fad66a6ce8f996908195593807c4b38bd528d2cff09d4eb3,
            ...   0x3e5bfbbf4d3e39b1a2fd816a7680c19ebebaf3a141b239934ad43cb33fcec8ce)
            >>> Account.recover_message(message, vrs=vrs)
            '0x5ce9454909639D2D17A3F753ce7d93fa0b9aB12E'

            >>> vrs = (
            ...   b'\x1c',
            ...   b'\xe6\xca\x9b\xbaX\xc8\x86\x11\xfa\xd6jl\xe8\xf9\x96\x90\x81\x95Y8\x07\xc4\xb3\x8b\xd5(\xd2\xcf\xf0\x9dN\xb3',
            ...   b'>[\xfb\xbfM>9\xb1\xa2\xfd\x81jv\x80\xc1\x9e\xbe\xba\xf3\xa1A\xb29\x93J\xd4<\xb3?\xce\xc8\xce')
            >>> Account.recover_message(message, vrs=vrs)
            '0x5ce9454909639D2D17A3F753ce7d93fa0b9aB12E'

            # variations on signature
            >>> signature = '0xe6ca9bba58c88611fad66a6ce8f996908195593807c4b38bd528d2cff09d4eb33e5bfbbf4d3e39b1a2fd816a7680c19ebebaf3a141b239934ad43cb33fcec8ce1c'
            >>> Account.recover_message(message, signature=signature)
            '0x5ce9454909639D2D17A3F753ce7d93fa0b9aB12E'
            >>> signature = b'\xe6\xca\x9b\xbaX\xc8\x86\x11\xfa\xd6jl\xe8\xf9\x96\x90\x81\x95Y8\x07\xc4\xb3\x8b\xd5(\xd2\xcf\xf0\x9dN\xb3>[\xfb\xbfM>9\xb1\xa2\xfd\x81jv\x80\xc1\x9e\xbe\xba\xf3\xa1A\xb29\x93J\xd4<\xb3?\xce\xc8\xce\x1c'
            >>> Account.recover_message(message, signature=signature)
            '0x5ce9454909639D2D17A3F753ce7d93fa0b9aB12E'
            >>> # Caution about this approach: likely problems if there are leading 0s
            >>> signature = 0xe6ca9bba58c88611fad66a6ce8f996908195593807c4b38bd528d2cff09d4eb33e5bfbbf4d3e39b1a2fd816a7680c19ebebaf3a141b239934ad43cb33fcec8ce1c
            >>> Account.recover_message(message, signature=signature)
            '0x5ce9454909639D2D17A3F753ce7d93fa0b9aB12E'
        """  # noqa: E501
        message_hash = _hash_eip191_message(signable_message)
        return cast(ChecksumAddress, self._recover_hash(message_hash, vrs, signature))

    @combomethod
    def _recover_hash(
        self,
        message_hash: Hash32,
        vrs: Optional[Tuple[VRS, VRS, VRS]] = None,
        signature: bytes = None,
    ) -> ChecksumAddress:
        hash_bytes = HexBytes(message_hash)
        if len(hash_bytes) != 32:
            raise ValueError("The message hash must be exactly 32-bytes")
        if vrs is not None:
            v, r, s = map(hexstr_if_str(to_int), vrs)
            v_standard = to_standard_v(v)
            signature_obj = self._keys.Signature(vrs=(v_standard, r, s))
        elif signature is not None:
            signature_bytes = HexBytes(signature)
            signature_bytes_standard = to_standard_signature_bytes(signature_bytes)
            signature_obj = self._keys.Signature(
                signature_bytes=signature_bytes_standard
            )
        else:
            raise TypeError("You must supply the vrs tuple or the signature bytes")
        pubkey = signature_obj.recover_public_key_from_msg_hash(hash_bytes)
        return cast(ChecksumAddress, pubkey.to_checksum_address())

    @combomethod
    def recover_transaction(self, serialized_transaction):
        """
        Get the address of the account that signed this transaction.

        :param serialized_transaction: the complete signed transaction
        :type serialized_transaction: hex str, bytes or int
        :returns: address of signer, hex-encoded & checksummed
        :rtype: str

        .. doctest:: python

            >>> raw_transaction = '0xf86a8086d55698372431831e848094f0109fc8df283027b6285cc889f5aa624eac1f55843b9aca008025a009ebb6ca057a0535d6186462bc0b465b561c94a295bdb0621fc19208ab149a9ca0440ffd775ce91a833ab410777204d5341a6f9fa91216a6f3ee2c051fea6a0428'
            >>> Account.recover_transaction(raw_transaction)
            '0x2c7536E3605D9C16a7a3D7b1898e529396a65c23'
        """  # noqa: E501
        txn_bytes = HexBytes(serialized_transaction)
        if len(txn_bytes) > 0 and txn_bytes[0] <= 0x7F:
            # We are dealing with a typed transaction.
            typed_transaction = TypedTransaction.from_bytes(txn_bytes)
            msg_hash = typed_transaction.hash()
            vrs = typed_transaction.vrs()
            return self._recover_hash(msg_hash, vrs=vrs)

        txn = Transaction.from_bytes(txn_bytes)
        msg_hash = hash_of_signed_transaction(txn)
        return self._recover_hash(msg_hash, vrs=vrs_from(txn))


    @combomethod
    def sign_message(
        self,
        signable_message: SignableMessage,
        private_key: Union[bytes, HexStr, int, keys.PrivateKey],
    ) -> SignedMessage:
        r"""
        Sign the provided message.

        This API supports any messaging format that will encode to EIP-191 messages.

        If you would like historical compatibility with :meth:`w3.eth.sign() <web3.eth.Eth.sign>`
        you can use :meth:`~eth_account.messages.encode_defunct`.

        Other options are the "validator", or "structured data" standards.
        You can import all supported message encoders in
        ``eth_account.messages``.

        :param signable_message: the encoded message for signing
        :param private_key: the key to sign the message with
        :type private_key: hex str, bytes, int or :class:`eth_keys.datatypes.PrivateKey`
        :returns: Various details about the signature - most importantly the
            fields: v, r, and s
        :rtype: ~eth_account.datastructures.SignedMessage

        .. doctest:: python

            >>> msg = "I♥SF"
            >>> from eth_account.messages import encode_defunct
            >>> msghash = encode_defunct(text=msg)
            >>> msghash
            SignableMessage(version=b'E',
             header=b'thereum Signed Message:\n6',
             body=b'I\xe2\x99\xa5SF')
            >>> # If you're curious about the internal fields of SignableMessage, take a look at EIP-191, linked above
            >>> key = "0xb25c7db31feed9122727bf0939dc769a96564b2de4c4726d035b36ecf1e5b364"
            >>> Account.sign_message(msghash, key)
            SignedMessage(messageHash=HexBytes('0x1476abb745d423bf09273f1afd887d951181d25adc66c4834a70491911b7f750'),
             r=104389933075820307925104709181714897380569894203213074526835978196648170704563,
             s=28205917190874851400050446352651915501321657673772411533993420917949420456142,
             v=28,
             signature=HexBytes('0xe6ca9bba58c88611fad66a6ce8f996908195593807c4b38bd528d2cff09d4eb33e5bfbbf4d3e39b1a2fd816a7680c19ebebaf3a141b239934ad43cb33fcec8ce1c'))



        .. _EIP-191: https://eips.ethereum.org/EIPS/eip-191
        """  # noqa: E501
        message_hash = _hash_eip191_message(signable_message)
        return cast(SignedMessage, self._sign_hash(message_hash, private_key))

    @combomethod
    def signHash(self, message_hash, private_key):
        """
        Sign the provided hash.

        .. WARNING:: *Never* sign a hash that you didn't generate,
            it can be an arbitrary transaction. For example, it might
            send all of your account's ether to an attacker.
            Instead, prefer :meth:`~eth_account.account.Account.sign_message`,
            which cannot accidentally sign a transaction.

        .. CAUTION:: Deprecated for :meth:`~eth_account.account.Account.sign_message`.
            This method will be removed in v0.6

        :param message_hash: the 32-byte message hash to be signed
        :type message_hash: hex str, bytes or int
        :param private_key: the key to sign the message with
        :type private_key: hex str, bytes, int or :class:`eth_keys.datatypes.PrivateKey`
        :returns: Various details about the signature - most
          importantly the fields: v, r, and s
        :rtype: ~eth_account.datastructures.SignedMessage
        """
        warnings.warn(
            "signHash is deprecated in favor of sign_message",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self._sign_hash(message_hash, private_key)

    @combomethod
    def _sign_hash(
        self,
        message_hash: Hash32,
        private_key: Union[bytes, HexStr, int, keys.PrivateKey],
    ) -> SignedMessage:
        msg_hash_bytes = HexBytes(message_hash)
        if len(msg_hash_bytes) != 32:
            raise ValueError("The message hash must be exactly 32-bytes")

        key = self._parsePrivateKey(private_key)

        (v, r, s, eth_signature_bytes) = sign_message_hash(key, msg_hash_bytes)
        return SignedMessage(
            messageHash=msg_hash_bytes,
            r=r,
            s=s,
            v=v,
            signature=HexBytes(eth_signature_bytes),
        )

    @combomethod
    def sign_transaction(self, transaction_dict, private_key):
        """
        Sign a transaction using a local private key.

        It produces signature details and the hex-encoded transaction suitable for
        broadcast using :meth:`w3.eth.sendRawTransaction()
        <web3.eth.Eth.sendRawTransaction>`.

        To create the transaction dict that calls a contract, use contract object:
        `my_contract.functions.my_function().buildTransaction()
        <http://web3py.readthedocs.io/en/latest/contracts.html#methods>`_

        Note: For non-legacy (typed) transactions, if the transaction type is not
        explicitly provided, it may be determined from the transaction parameters of
        a well-formed transaction. See below for examples on how to sign with
        different transaction types.

        :param dict transaction_dict: the transaction with available keys, depending
          on the type of transaction: nonce, chainId, to, data, value, gas, gasPrice,
          type, accessList, maxFeePerGas, and maxPriorityFeePerGas
        :param private_key: the private key to sign the data with
        :type private_key: hex str, bytes, int or :class:`eth_keys.datatypes.PrivateKey`
        :returns: Various details about the signature - most
          importantly the fields: v, r, and s
        :rtype: AttributeDict
        """
        if not isinstance(transaction_dict, Mapping):
            raise TypeError(
                "transaction_dict must be dict-like, got %r" % transaction_dict
            )

        account = self.from_key(private_key)

        # allow from field, *only* if it matches the private key
        if "from" in transaction_dict:
            if transaction_dict["from"] == account.address:
                sanitized_transaction = dissoc(transaction_dict, "from")
            else:
                raise TypeError(
                    "from field must match key's %s, but it was %s"
                    % (
                        account.address,
                        transaction_dict["from"],
                    )
                )
        else:
            sanitized_transaction = transaction_dict

        # sign transaction
        (
            v,
            r,
            s,
            encoded_transaction,
        ) = sign_transaction_dict(account._key_obj, sanitized_transaction)
        transaction_hash = keccak(encoded_transaction)

        return SignedTransaction(
            rawTransaction=HexBytes(encoded_transaction),
            hash=HexBytes(transaction_hash),
            r=r,
            s=s,
            v=v,
        )

    @combomethod
    def _parsePrivateKey(self, key):
        """
        Generate a :class:`eth_keys.datatypes.PrivateKey` from the provided key.

        If the key is already of type :class:`eth_keys.datatypes.PrivateKey`,
        return the key.

        :param key: the private key from which a :class:`eth_keys.datatypes.PrivateKey`
                    will be generated
        :type key: hex str, bytes, int or :class:`eth_keys.datatypes.PrivateKey`
        :returns: the provided key represented as a
                  :class:`eth_keys.datatypes.PrivateKey`
        """
        if isinstance(key, self._keys.PrivateKey):
            return key

        try:
            return self._keys.PrivateKey(HexBytes(key))
        except ValidationError as original_exception:
            raise ValueError(
                "The private key must be exactly 32 bytes long, instead of "
                "%d bytes." % len(key)
            ) from original_exception

    @combomethod
    def sign_typed_data(
        self,
        private_key: Union[bytes, HexStr, int, keys.PrivateKey],
        domain_data: Dict[str, Any] = None,
        message_types: Dict[str, Any] = None,
        message_data: Dict[str, Any] = None,
        full_message: Dict[str, Any] = None,
    ) -> SignedMessage:
        r"""
        Sign the provided EIP-712 message with the provided key.

        :param private_key: the key to sign the message with
        :param domain_data: EIP712 domain data
        :param message_types: custom types used by the `value` data
        :param message_data: data to be signed
        :param full_message: a dict containing all data and types
        :type private_key: hex str, bytes, int or :class:`eth_keys.datatypes.PrivateKey`
        :type domain_data: dict
        :type message_types: dict
        :type message_data: dict
        :type full_message: dict
        :returns: Various details about the signature - most importantly the
            fields: v, r, and s
        :rtype: ~eth_account.datastructures.SignedMessage
        """  # noqa: E501
        signable_message = encode_typed_data(
            domain_data,
            message_types,
            message_data,
            full_message,
        )
        message_hash = _hash_eip191_message(signable_message)
        return cast(SignedMessage, self._sign_hash(message_hash, private_key))
