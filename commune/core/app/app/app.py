import commune as c
import os

class App:

    def forward(self, public=False, port=3000, api_port=8000):
        return {
            "api": self.api(port=api_port, free_mode=True),
            "app": self.app(public=public),

        }
        
    def app(self, port=3000, public=False, remote=True, build=True, api_port=8000, api_free_mode=True):
        if not c.server_exists('api'):
            print('API not found, please run `commune api` first.')
            self.api(port=api_port, free_mode=api_free_mode)
        api_ip = c.ip() if public else '0.0.0.0'
        api_url = f'http://{api_ip}:{api_port}'
        cwd = c.dirpath('app') 
        params = {
            'name': 'app', 
            'build': {'context': './'},
            'port':  port,
            'env': {'API_URL': api_url},
            'volumes': [f'{cwd}:/app','/app/node_modules'],
            'cwd': cwd  ,
            'working_dir': '/app',
            # 'cmd': 'npm start',
            'daemon': remote,
        }
        return c.fn('pm/run')(**params)



    def api(self, port=8000, free_mode=True):   
        return c.serve('api', port=port, free_mode=free_mode)
        