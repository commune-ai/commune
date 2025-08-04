import commune as c
import os

class App:

    def forward(self, public=True, port=3000):
        return {
            "api": self.api( port=8000, free_mode=True),
            "app": self.app(),

        }
        
    def app(self, port=3000, public=True, remote=True, build=True):
        self.api()
        dirpath = c.dirpath('app')        
        params = {
            'build': True,
            'port':  port,
            'env': {'API_URL': f'http://localhost:8000'},
            'volumes': [f'{dirpath}:/app','/app/node_modules'],
            'name': 'app', 
            'cwd': dirpath,
            'working_dir': '/app',
        }
        return c.fn('pm/run')(**params, daemon=remote)

    def api(self, port=8000, free_mode=True):   
        return c.serve('api', port=port, free_mode=free_mode)
        