class Tool:
    description = "This module is used to find files and modules in the current directory"
    def forward(self, path='./', **kwargs):
        return self.reduce(str(self.file2text(path=path), **kwargs))
    