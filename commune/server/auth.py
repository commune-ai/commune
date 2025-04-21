import base64
import hmac
import json
import time
from typing import Dict, Optional, Any

import commune as c

class JWT:

    def get_headers(self, data: Any, key:str=None, crypto_type='ecdsa', mode='dict') -> dict:
        """
        Generate the headers with the JWT token
        """
        headers =  self.get_token(c.hash(data), key=key, crypto_type=crypto_type, mode=mode)
        return headers

    def verify_headers(self, headers: str, data:Optional[Any]=None) -> Dict:
        """
        Verify and decode a JWT token
        """
        verified = self.verify_token(headers['token'])
        assert verified, 'Invalid signature'
        if data != None:
            assert verified['data'] == c.hash(data), 'Invalid data {} != {}'.format(verified['data'], c.hash(data))
        return verified

    def check_crypto_type(self, crypto_type):
        assert crypto_type in ['ecdsa', 'sr25519'], f'Invalid crypto_type {crypto_type}'

    def get_token(self, data: Dict='hey',  key:Optional[str]=None,   crypto_type: str = 'ecdsa', expiration: int = 3600, mode='bytes') -> str:
        """
        Generate a JWT token with the given data
        Args:
            data: Dictionary containing the data to encode in the token
            expiration: Optional custom expiration time in seconds
        Returns:
            JWT token string
        """
        if isinstance(key, str) or key == None:
            key = c.get_key(key, crypto_type=crypto_type)
        else:
            key = key
            if crypto_type != key.get_crypto_type(key.crypto_type):
                crypto_type = key.get_crypto_type(key.crypto_type)

        self.check_crypto_type(crypto_type)
        if not isinstance(data, dict):
            data = {'data': data }
        token_data = data.copy()        
        # Add standard JWT claims
        token_data.update({
            'iat': str(float(c.time())),  # Issued at time
            'exp': str(float(c.time() + expiration)),  # Expiration time
            'iss': key.key_address,  # Issuer (key address)
        })
        header = {
            'alg': crypto_type,
            'typ': 'JWT',
        }
        # Create message to sign
        message = f"{self._base64url_encode(header)}.{self._base64url_encode(token_data)}"
        # For asymmetric algorithms, use the key's sign method
        signature = key.sign(message, mode='bytes')
        signature_encoded = self._base64url_encode(signature)
        # Combine to create the token
        token = f"{message}.{signature_encoded}"
        if mode == 'dict':
            return self.verify_token(token)
        elif mode == 'bytes':
            return f"{message}.{signature_encoded}"
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'bytes' or 'dict'.")
            
    def verify_token(self, token: str) -> Dict:
        """
        Verify and decode a JWT token
        """
        if isinstance(token, dict) and 'token' in token:
            token = token['token']
        # Split the token into parts
        header_encoded, data_encoded, signature_encoded = token.split('.')
        # Decode the data
        data = json.loads(self._base64url_decode(data_encoded))
        headers = json.loads(self._base64url_decode(header_encoded))
        # Check if token is expired
        if 'exp' in data and float(data['exp']) < c.time():
            raise Exception("Token has expired")
        # Verify signature
        message = f"{header_encoded}.{data_encoded}"
        signature = self._base64url_decode(signature_encoded)
        assert c.verify(data=message, signature=signature, address=data['iss'], crypto_type=headers['alg']), "Invalid token signature"
        # data['data'] = message
        data['time'] = data['iat'] # set time field for semanitcally easy people
        data['signature'] = '0x'+signature.hex()
        data['alg'] = headers['alg']
        data['typ'] = headers['typ']
        data['token'] = token
        data['key'] = data['iss']
        return data

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

    def test_token(self, test_data = {'fam': 'fam', 'admin': 1} , crypto_type='ecdsa'):
        """
        Test the JWT token functionality
        
        Returns:
            Dictionary with test results
        """
        # Generate a token
        token = self.get_token(test_data, crypto_type=crypto_type)
        # Verify the token
        decoded = self.verify_token(token)
        # Check if original data is in the decoded data
        validation_passed = all(test_data[key] == decoded[key] for key in test_data)
        assert validation_passed, "Decoded data does not match original data"
        # Test token expiration
        quick_token = self.get_token(test_data, expiration=0.1, crypto_type=crypto_type)
        time.sleep(0.2)  # Wait for token to expire
        
        expired_token_caught = False
        try:
            decoded = self.verify_token(quick_token)
        except Exception as e:
            expired_token_caught = True
        assert expired_token_caught, "Expired token not caught"
        
        return {
            "token": token,
            "decoded_data": decoded,
            "crypto_type": crypto_type,
            "quick_token": quick_token,
            "expired_token_caught": expired_token_caught
            }

    def test_headers(self, key='test.jwt', crypto_type='ecdsa'):
        data = {'fn': 'test', 'params': {'a': 1, 'b': 2}}
        headers = self.get_headers(data, key=key, crypto_type=crypto_type)
        verified = self.verify_headers(headers)
        verified = self.verify_headers(headers, data=data)
        return {'headers': headers, 'verified': verified}

    def test(self):
        return {
            'token': self.test_token(),
            'headers': self.test_headers()
        }