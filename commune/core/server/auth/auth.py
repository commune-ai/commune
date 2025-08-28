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
                signature_keys = ['data', 'time', 'cost']):
        
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

    def forward(self,  data: Any,  key=None, crypto_type=None, cost=0) -> dict:
        """
        Generate the headers with the JWT token
        """
        key = self.get_key(key, crypto_type=crypto_type)
        result = {
            'data': self.hash(data),
            'time': str(time.time()),
            'key': key.key_address,
            'cost': str(cost)
        }
        result['signature'] = key.sign(self.get_sig_data(result), mode='str')
        return result

    headers = generate = forward


    def get_sig_data(self, headers: Dict[str, str]) -> str:
        assert all(k in headers for k in self.signature_keys), f'Missing keys in headers {headers}'
        return json.dumps({k: headers[k] for k in self.signature_keys}, separators=(',', ':'))

    def verify(self, headers: str, data:Optional[Any]=None, max_age=1000) -> bool:
        """
        Verify and decode a JWT token
        provide the data if you want to verify the data hash
        """

        # check age 
        crypto_type = headers.get('crypto_type', self.crypto_type)
        age = abs(time.time() - float(headers['time']))
        max_age = max_age or self.max_age
        assert age < max_age, f'Token is stale {age} > {max_age}'
        assert 'signature' in headers, 'Missing signature'
        sig_data = self.get_sig_data(headers)
        verified = self.key.verify(sig_data, signature=headers['signature'], address=headers['key'], crypto_type=crypto_type)
        assert verified, f'Invalid signature {headers}'
        if data != None:
            assert headers['data'] == self.hash(data), f'Invalid data {data}'
        return verified

    verify_headers = verify


    def _is_identity_hash_type(self):
        return self.hash_type in ['identity', None, 'none']

    def hash(self, data: Any) -> str:
        """
        Hash the data using sha256
        """
        data = json.dumps(data, separators=(',', ':'))
        if self.hash_type == 'sha256':
            return c.hash(data)
        elif self._is_identity_hash_type():
            return data
        else: 
            raise ValueError(f'Invalid hash type {self.hash_type}')

    def reverse_hash(self, data: str) -> Any:
        """
        Reverse the hash to get the original data
        This is not possible for sha256, so we return the data as is
        """
        data = json.loads(data, separators=(',', ':'))
        if self.hash_type in ['identity', None, 'none']:
            return 
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
        assert auth.verify(headers)
        return {'headers': headers}


    def sand(self):

        headers = {
  "data": "77c7d662fd19f9c6844796d3c909ae727fb4b500e17184af20e895c1f8dc7b3b",
  "time": "1756312652.381",
  "key": "5FxnwR1rJ3yzHwRPVNgBjG79J9q7LVTUy6suePFjx9UfNpaC",
  "signature": "0xae9b1eac55d27f36fd1e225af4fcc775e45ffadc5b8657a2d61457e206987f7ce0e1f5fc47c2e34382a6d185039096c51e4495304d17fd7a629ccd3065001984",
  "hash_type": "sha256",
  "crypto_type": "sr25519",
  "cost": "10",
  "sigData": "{\"data\":\"77c7d662fd19f9c6844796d3c909ae727fb4b500e17184af20e895c1f8dc7b3b\",\"time\":\"1756312652.381\",\"cost\":\"10\"}",
#   "verified": true
}
        sigDataRecontructed = json.dumps({k: headers[k] for k in ['data', 'time', 'cost']})
        # assert sigDataRecontructed == headers['sigData'], f'sigData does not

        # reconstruct the signed data
        sigDataRecontructed = json.dumps({k: headers[k] for k in ['data', 'time', 'cost']}, separators=(',', ':'))
        assert c.hash(sigDataRecontructed) == c.hash(headers['sigData']), f'sigData does not match {sigDataRecontructed} != {headers["sigData"]}'
        # return sigDataRecontructed
        return c.verify(headers['sigData'], headers['signature'], headers['key'])