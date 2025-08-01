import commune as c
import os

class App:

    def forward(self):
        return {
            "api": self.api(),
            "app": self.app(),

        }
        
    def app(self, port=3000):
        self.api()
        cwd = c.dirpath('app')
        os.system( f'cd {cwd} && docker compose up -d')
        return {"status": "app started", 'cwd': cwd, 'url': f'http://localhost:{port}'}

    def api(self, port=8000, free_mode=True):   
        return c.serve('api', port=port, free_mode=free_mode)
        