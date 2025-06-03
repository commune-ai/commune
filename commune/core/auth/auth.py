import base64
import hmac
import json
import time
from typing import Dict, Optional, Any
import commune as c

class Auth:

    def __init__(self, key=None, 
                crypto_type='sr25519', 
                hash_type='sha256',    
                max_staleness=60, 
                signature_keys = ['data', 'time']):
        
        """

        Initialize the Auth class
        :param key: the key to use for signing
        :param crypto_type: the crypto type to use for signing
        :param hash_type: the hash type to use for signing
        :param signature_keys: the keys to use for signing 
        """
        self.signature_keys = signature_keys
        self.key = c.get_key(key, crypto_type=crypto_type)
        self.hash_type = hash_type
        self.crypto_type = crypto_type
        self.signature_keys = signature_keys
        self.max_staleness = max_staleness

    def headers(self,  data: Any,  key=None, crypto_type=None, signature_keys=None) -> dict:
        """
        Generate the headers with the JWT token
        """
        key = self.get_key(key, crypto_type=crypto_type)
        signature_keys = signature_keys or self.signature_keys
        result = {
            'data': self.hash(data),
            'time': str(time.time()),
            'key': key.key_address,
        }
        result['signature'] = key.sign({k: result[k] for k in signature_keys}, mode='str')
        return result

    get_headers = headers

    def verify(self, headers: str, data:Optional[Any]=None) -> Dict:
        """
        Verify and decode a JWT token
        provide the data if you want to verify the data hash
        """

        # check time 

        time_now = time.time()
        crypto_type = headers.get('crypto_type', self.crypto_type)
        signature_keys = headers.get('signature_keys', self.signature_keys)

        staleness = abs(time_now - float(headers['time']))
        assert staleness < self.max_staleness, f'Token is stale {staleness} > {self.max_staleness}'
        assert 'signature' in headers, 'Missing signature'
        sig_data = {k: headers[k] for k in signature_keys}
        verified = self.key.verify(sig_data, signature=headers['signature'], address=headers['key'], crypto_type=crypto_type)
        assert verified, 'Invalid signature'
        if data != None:
            rehash_data = self.hash(data)
            assert headers['data'] == rehash_data, f'Invalid data {headers["data"]} != {rehash_data}'
        return headers

    verify_headers = verify

    def hash(self, data: Any) -> str:
        """
        Hash the data using sha256
        """
        if self.hash_type == 'sha256':
            if isinstance(data, str):
                data = data.encode('utf-8')
            elif isinstance(data, dict):
                data = json.dumps(data)
            return c.hash(data)
        else: 
            raise ValueError(f'Invalid hash type {self.hash_type}')

    def get_crypto_type(self, crypto_type=None):
        """
        Get the crypto type
        """
        if crypto_type is None:
            crypto_type = self.crypto_type
        assert crypto_type in ['sr25519', 'ed25519'], f'Invalid crypto type {crypto_type}'
        return crypto_type

    def get_key(self, key=None, crypto_type=None):
        crypto_type = self.get_crypto_type(crypto_type)
        if key is None:
            key = self.key
        if isinstance(key, str):
            key = c.get_key(key, crypto_type=crypto_type)
        assert hasattr(key, 'key_address'), f'Invalid key {key}'
        return key


    def test(self, key='test.auth', crypto_type='sr25519'):
        data = {'fn': 'test', 'params': {'a': 1, 'b': 2}}
        auth = Auth(key=key, crypto_type=crypto_type)
        headers = auth.headers(data, key=key, crypto_type=crypto_type)
        headers = auth.verify(headers)
        headers = auth.verify(headers, data=data)
        return {'headers': headers}
