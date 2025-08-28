import commune as c
import os

class App:




    def serve(self, public=False, port=3000, api_port=8000):
        return {
            "api": self.api(port=api_port),
            "app": self.app(public=public),

        }

    forward = serve  # Alias for serve method
        
    def app(self, port=3000, public=False, remote=True, build=True, api_port=8000):
        if not c.server_exists('api'):
            print('API not found, please run `commune api` first.')
            self.api(port=api_port)
        api_ip = c.ip() if public else '0.0.0.0'
        api_url = f'http://{api_ip}:{api_port}'
        cwd = c.dirpath('app') 
        params = {
            'name': 'app', 
            'build': {'context': './'},
            'port':  port,
            'env': {'API_URL': api_url, 'APP_URL': f'http://0.0.0.0:{port}'},
            'volumes': [f'{cwd}:/app','/app/node_modules'],
            'cwd': cwd  ,
            'working_dir': '/app',
            # 'cmd': 'npm start',
            'daemon': remote,
        }
        return c.fn('pm/run')(**params)

    def api(self, port=8000):   
        return c.serve('api', port=port)
        


    def sand(self):
        import json
        x= {
  "data": "77c7d662fd19f9c6844796d3c909ae727fb4b500e17184af20e895c1f8dc7b3b",
  "time": "1754782711.944",
  "key": "5EsrtnVcfGghopykFDiPbhS1qs6J9evUtZLLbcBZgGBnMkig",
  "signature": "0x426304b17e37001915fe948c91e917ce38265a1d64ee5591c952030be106e14e9864b00a135e46d32ea632a074105111970ed4d4072948a379a2520b861d008f",
  "hash_type": "sha256",
  "crypto_type": "sr25519",
  "data_hash": "851b158c61343ccf251a7bfa92bff4b3f03d5261e3f1faa6cc6aa0cdfdc29061",
  "verified": True
}
        data = json.dumps({"data": x['data'], "time": x['time']})
        data_hash = c.hash(data)
        assert x['data_hash'] == data_hash, f"Data hash mismatch: {x['data_hash']} != {data_hash}, data: {data}"
        auth = c.mod('auth')()
        return auth.verify(x, max_age=60000)
