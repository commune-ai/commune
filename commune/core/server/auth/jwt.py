import base64
import hmac
import json
import time
from typing import Dict, Optional, Any
import commune as c

class AuthJWT:
    description = 'auth'

    def __init__(self, key=None, crypto_type: str = 'sr25519'):
        self.key = c.get_key(key, crypto_type=crypto_type)

    @property
    def crypto_type(self) -> str:
        return self.key.crypto_type_name
        
    def generate(self, data: Any, key:str=None, mode='headers') -> dict:
        """
        Generate the headers with the JWT token
        """
        headers =  self.token(c.hash(data), key=key, mode=mode)
        return headers

    headers = forward = generate

    def hash(self, data: Any) -> str:
        """
        Hash the data using sha256
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        elif isinstance(data, dict):
            data = json.dumps(data)
        return c.hash(data)

    def get_key(self, key) -> Any:
        if key == None:
            return self.key
        else:
            key = c.get_key(key, crypto_type=self.crypto_type)
        assert key.crypto_type_name == self.crypto_type, f"Key crypto type {key.crypto_type} does not match expected {self.crypto_type}"
        return key
        
    def token(self, data: Dict='hey',  key:Optional[str]=None, expiration: int = 3600, mode='bytes') -> str:
        """
        Generate a JWT token with the given data
        Args:
            data: Dictionary containing the data to encode in the token
            expiration: Optional custom expiration time in seconds
            mode: 'bytes' to return as string, 'dict'/'headers' to return as dictionary with metadata
        Returns:
            JWT token string
        """
        key = self.get_key(key)
        token_data = {
            'data': data,
            'iat': str(float(c.time())),  # Issued at time
            'exp': str(float(c.time() + expiration)),  # Expiration time
            'iss': key.key_address,  # Issuer (key address)
        }
        header = {
            'alg': self.crypto_type,
            'typ': 'JWT',
        }
        # Create message to sign
        message = f"{self._base64url_encode(header)}.{self._base64url_encode(token_data)}"
        # For asymmetric algorithms, use the key's sign method
        signature = self._base64url_encode(key.sign(message, mode='bytes'))
        # Combine to create the token
        token = f"{message}.{signature}"
        if mode in ['dict', 'headers']:
            return {
                'token': token,
                'time': token_data['iat'],
                'exp': token_data['exp'],
                'key': key.key_address,
                'alg': header['alg'],
                'typ': header['typ'],
            }
        elif mode == 'bytes':
            return f"{message}.{signature}"
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'bytes' or 'dict'.")


    def is_headers(self, token: str) -> bool:
        """
        Check if the token is in headers format (dict with 'token' key)
        """
        return isinstance(token, dict) and 'token' in token

            
    def verify(self, token: str) -> Dict:
        """
        Verify and decode a JWT token
        """
        if self.is_headers(token):
            token = token['token']
        # Split the token into parts
        header_encoded, data_encoded, signature_encoded = token.split('.')
        # Decode the data
        data = json.loads(self._base64url_decode(data_encoded))
        headers = json.loads(self._base64url_decode(header_encoded))
        # Check if token is expired
        if 'exp' in data and float(data['exp']) < c.time():
            raise Exception("Token has expired")
        message = f"{header_encoded}.{data_encoded}"
        signature = self._base64url_decode(signature_encoded)
        assert self.key.verify(data=message, signature=signature, address=data['iss'], crypto_type=headers['alg']), "Invalid token signature"
        return True

    def _base64url_encode(self, data):
        """Encode data in base64url format"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        elif isinstance(data, dict):
            data = json.dumps(data, separators=(',', ':')).encode('utf-8')
        encoded = base64.urlsafe_b64encode(data).rstrip(b'=')
        return encoded.decode('utf-8')
    
    def _base64url_decode(self, data):
        """Decode base64url data"""
        padding = b'=' * (4 - (len(data) % 4))
        return base64.urlsafe_b64decode(data.encode('utf-8') + padding)

    def test_token(self, test_data = {'fam': 'fam', 'admin': 1}):
        """
        Test the JWT token functionality
        
        Returns:
            Dictionary with test results
        """
        # Generate a token
        token = self.token(test_data)
        # Verify the token
        assert self.verify(token)
        # Test token expiration
        quick_token = self.token(test_data, expiration=0.1)
        time.sleep(0.2)  # Wait for token to expire
        
        expired_token_caught = False
        try:
            decoded = self.verify(quick_token)
        except Exception as e:
            expired_token_caught = True
        assert expired_token_caught, "Expired token not caught"
        
        return {
            "token": token,
            "crypto_type": self.crypto_type,
            "quick_token": quick_token,
            "expired_token_caught": expired_token_caught
            }

    def test_headers(self, key='test.jwt'):
        data = {'fn': 'test', 'params': {'a': 1, 'b': 2}}
        headers = self.generate(data, key=key)
        verified = self.verify(headers)
        verified = self.verify(headers)
        return {'headers': headers, 'verified': verified}

    def test(self):
        crypto_types = ['sr25519', 'ed25519']
        result = {}
        for crypto_type in crypto_types:
            self.key = c.get_key('test.jwt', crypto_type=self.crypto_type)
            result[crypto_type] = {
                'token': self.test_token(),
                'headers': self.test_headers()
            }
            print(f"Tested JWT with crypto_type {crypto_type}: {result}")
        
        return result