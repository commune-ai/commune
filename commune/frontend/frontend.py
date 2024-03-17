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


    def logs(self):
        c.cmd('docker logs -f frontend.commune.v0')

    def docs_path(self):
        return self.frontend_path + '/docs'

    def doc_modules_path(self):
        return self.frontend_path + '/docs/modules'
    def docs(self):
        return c.ls(self.docs_path())


    def copy_docs(self):
        docs_path = self.docs_path()
        module2docpath = c.module2docpath()

        for module,module_doc_path in module2docpath.items():
            frontend_module_doc_path = self.doc_modules_path() + '/' + module + '.md'
            c.cp(module_doc_path, frontend_module_doc_path, refresh=True)
            assert c.exists(frontend_module_doc_path)
            c.print('Copied docs for module: ' + module)

        return c.ls(docs_path)
        



