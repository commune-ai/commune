import commune as c
import time

class Auth:

    feautres=['key', 'time', 'data', 'signature']
    def verify_headers(self, headers:dict, data=None, max_staleness=10):

        headers = c.copy({k: headers[k] for k in self.feautres})
        headers['time'] = float(headers.get('time', c.time()))
        headers = self.order_dict(headers)
        signature = headers.pop('signature')
        print('headers', headers)
        if data:
            assert c.hash(data) == headers['data'], f'InvalidDataHash({c.hash(data)} != {headers["data"]})'
        staleness = c.time() - float(headers['time'])
        assert isinstance(headers, dict), f'Headers must be a dict, not {type(headers)}'
        assert  staleness < max_staleness, f"Request is too old ({staleness}s > {max_staleness}s (MAX)" 
        print('headers', headers)
        c.verify(headers, signature, address=headers['key']), f'InvalidSignature'
        return headers

    def order_dict(self, d:dict):
        return {k: d[k] for k in sorted(d)}

    def get_headers(self, data, key=None, content_type='application/json'):
        data_hash = c.hash(data)
        key = c.get_key(key)
        headers = {'key': key.ss58_address, 'time': str(time.time()), 'data': c.hash(data), 'content_type': content_type}
        headers['signature'] = c.sign(headers, key=key, mode='str')
        return headers

    def test(self):
        data = {'fn': 'test', 'params': {'a': 1, 'b': 2}}
        headers = self.get_headers(data)
        verified = self.verify_headers(headers)
        verified = self.verify_headers(headers, data)
        return {'headers': headers, 'verified': verified}