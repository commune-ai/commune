import commune as c
import os

class App:

    def forward(self, api_public=True, port=3000, api_port=8000):
        return {
            "api": self.api(port=api_port, free_mode=True),
            "app": self.app(api_public=api_public),

        }
        
    def app(self, port=3000, public=False, remote=True, build=True, api_port=8000, api_free_mode=True, api_public=True):
        if not c.exists('api'):
            print('API not found, please run `commune api` first.')
            self.api(port=api_port, free_mode=api_free_mode)
        dirpath = c.dirpath('app')   
        api_ip = c.ip() if api_public else '0.0.0.0'
        api_url = f'http://{api_ip}:{api_port}'
        params = {
            'build': True,
            'port':  port,
            'env': {'API_URL': api_url},
            'volumes': [f'{dirpath}:/app','/app/node_modules'],
            'name': 'app', 
            'cwd': dirpath,
            'working_dir': '/app',
        }
        return c.fn('pm/run')(**params, daemon=remote)

    def api(self, port=8000, free_mode=True):   
        return c.serve('api', port=port, free_mode=free_mode)
        