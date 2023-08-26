import commune as c

class Frontend(c.Module):
    def __init__(self, **kwargs):
        config = self.set_config(config=kwargs)
        self.docker = c.module('docker')
    def run(self):
        print('Base run')

    frontend_path = c.repo_path + '/frontend'
    compose_path = frontend_path + '/docker-compose.yml'
    def up(self, port=300):
        c.compose(path=self.frontend_path + '/docker-compose.yml')

    
    def down(self):
        c.compose(path=self.frontend_path + '/docker-compose.yml', down=True)

    def docs_path(self):
        return self.frontend_path + '/docs'
    def docs(self):
        return c.ls(self.docs_path())



