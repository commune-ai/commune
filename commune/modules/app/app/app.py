import commune as c
import os

class App:

    api_port = 8000
    app_port = 3000

    def forward(self, public=False):
        return {
            "api": self.api(),
            "app": self.app(public=public),

        }

    serve = forward  # Alias for serve method
        
    def app(self, public=False, remote=True, build=True):
        cwd = c.dirpath('app') 
        ip = c.ip() if public else '0.0.0.0'
        api_url = f'http://{ip}:{self.api_port}'
        app_url = f'http://{ip}:{self.app_port}'
        params = {
            'name': 'app', 
            'build': {'context': './'},
            'port':  self.app_port,
            'env': {'NEXT_PUBLIC_API_URL': api_url, 
                    'NEXT_PUBLIC_APP_URL': app_url},
            'volumes': [f'{cwd}:/app','/app/node_modules'],
            'cwd': cwd  ,
            'working_dir': '/app',
            # 'cmd': 'npm start',
            'daemon': remote,
        }
        return c.fn('pm/run')(**params)

    def api(self):   
        return c.serve('api', port=self.api_port)
    

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
