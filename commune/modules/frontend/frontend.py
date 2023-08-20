import commune as c

class Frontend(c.Module):
    def __init__(self, **kwargs):
        config = self.set_config(config=kwargs)
        self.docker = c.module('docker')
    def run(self):
        print('Base run')

    frontend_path = c.repo_path + '/frontend'
    compose_path = frontend_path + '/docker-compose.yml'
    def build(self, port=300):
        c.compose(path=self.frontend_path + '/docker-compose.yml')



