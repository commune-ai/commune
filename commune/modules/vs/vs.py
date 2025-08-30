
import commune as c
import os
class Vscode:
    def __init__(self, lib_path = '~/commune'):
        self.lib_path = os.path.abspath(os.path.expanduser(lib_path))
    def forward(self, path = None, module=None):
        if module != None:
            path = c.dirpath(module)
        path = path or self.lib_path
        path = os.path.abspath(path)
        return self.cmd(f'code {path}')
    