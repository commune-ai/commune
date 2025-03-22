import commune as c
import time

class Auth:

    data_features=['data', 'time']

    def get_headers(self, data='fam', key=None, content_type='application/json', crypto_type='ecdsa'):
        key = c.get_key(key, crypto_type=crypto_type)
        headers_data = {'data': c.hash(data), 'time': str(time.time())}
        headers = {k: headers_data[k] for k in self.data_features}
        headers['signature'] = key.sign(headers_data, mode='str')
        headers['crypto_type'] = crypto_type
        headers['key'] = key.key_address
        return headers

    def verify_headers(self, headers:dict, data=None, max_staleness=10):
        signature = headers['signature']
        headers_data = {k: str(headers[k]) for k in self.data_features}
        if data:
            assert c.hash(data) == headers['data'], f'InvalidDataHash({c.hash(data)} != {headers["data"]})'
        staleness = c.time() - float(headers['time'])
        headers['time'] = float(headers['time'])
        assert isinstance(headers, dict), f'Headers must be a dict, not {type(headers)}'
        assert  staleness < max_staleness, f"Request is too old ({staleness}s > {max_staleness}s (MAX)" 
        assert c.verify(headers_data, signature, address=headers['key'], crypto_type=headers['crypto_type']), f'InvalidSignature'
        return headers


    def test_headers(self,crypto_type='sr25519'):
        results = {}
        for crypto_type in ['ecdsa', 'sr25519']:
            data = {'fn': 'test', 'params': {'a': 1, 'b': 2}}
            headers = self.get_headers(data, crypto_type=crypto_type)
            verified = self.verify_headers(headers)
            results[crypto_type] = {'headers': headers, 'verified': verified}
        return results




    def test(self,crypto_type='sr25519'):
        results = {}
        for crypto_type in ['ecdsa', 'sr25519']:
            data = {'fn': 'test', 'params': {'a': 1, 'b': 2}}
            headers = self.get_headers(data, crypto_type=crypto_type)
            verified = self.verify_headers(headers)
            results[crypto_type] = {'headers': headers, 'verified': verified}
        return results
        