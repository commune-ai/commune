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
                max_age=60, 
                signature_keys = ['data', 'time']):
        
        """

        Initialize the Auth class
        :param key: the key to use for signing
        :param crypto_type: the crypto type to use for signing
        :param hash_type: the hash type to use for signing
        :param signature_keys: the keys to use for signing 
        """
        self.signature_keys = signature_keys
        self.key = self.get_key(key, crypto_type=crypto_type)
        self.hash_type = hash_type
        self.crypto_type = crypto_type
        self.signature_keys = signature_keys
        self.max_age = max_age

    def forward(self,  data: Any,  key=None, crypto_type=None) -> dict:
        """
        Generate the headers with the JWT token
        """
        key = self.get_key(key, crypto_type=crypto_type)
        result = {
            'data': self.hash(data),
            'time': str(time.time()),
            'key': key.key_address,
        }
        result['signature'] = key.sign({k: result[k] for k in self.signature_keys}, mode='str')
        return result

    headers = generate = forward

    def verify(self, headers: str, data:Optional[Any]=None, max_age=1000) -> bool:
        """
        Verify and decode a JWT token
        provide the data if you want to verify the data hash
        """

        # check age 
        crypto_type = headers.get('crypto_type', self.crypto_type)
        age = abs(time.time() - float(headers['time']))
        max_age = max_age or self.max_age
        print(f'Age: {age}, Max Age: {max_age}')
        assert age < max_age, f'Token is stale {age} > {max_age}'
        assert 'signature' in headers, 'Missing signature'
        assert all(k in headers for k in self.signature_keys), f'Missing keys in headers {headers}'
        sig_data = json.dumps({k: headers[k] for k in self.signature_keys})
        verified = self.key.verify(sig_data, signature=headers['signature'], address=headers['key'], crypto_type=crypto_type)
        assert verified, f'Invalid signature {headers}'
        if data != None:
            rehash_data = self.hash(data)
            assert headers['data'] == rehash_data, f'Invalid data {headers["data"]} != {rehash_data}'
        return verified

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
        elif self.hash_type in ['identity', None, 'none']:
            return json.dumps(data)
        else: 
            raise ValueError(f'Invalid hash type {self.hash_type}')

    def reverse_hash(self, data: str) -> Any:
        """
        Reverse the hash to get the original data
        This is not possible for sha256, so we return the data as is
        """
        if self.hash_type in ['identity', None, 'none']:
            return json.loads(data)
        else:
            raise ValueError(f'Reverse hash not supported for {self.hash_type}')

    def get_key(self, key=None, crypto_type=None):
        crypto_type =  crypto_type or self.crypto_type
        if not hasattr(self, 'key'):
            self.key = c.get_key(key, crypto_type=crypto_type)
        if key is None:
            key = self.key
        else:
            key = c.get_key(key, crypto_type=crypto_type)
        assert hasattr(key, 'key_address'), f'Invalid key {key}'
        return key

    def test(self, key='test.auth', crypto_type='sr25519'):
        data = {'fn': 'test', 'params': {'a': 1, 'b': 2}}
        auth = Auth(key=key, crypto_type=crypto_type)
        headers = auth.forward(data, key=key, crypto_type=crypto_type)
        headers = auth.verify(headers)
        headers = auth.verify(headers, data=data)
        return {'headers': headers}
