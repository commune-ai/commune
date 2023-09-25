import commune as c

class Namespace(c.Module):

    def __init__(self, name = 'local'):
        self.name = name
    def add_namespace(self, name):
        return self.put(name, {})
    def exists(self, name):
        return self.exists(name)
    def rm_namespace(self, name):
        return self.rm(name)
    
    @classmethod
    def ls_namespaces(cls):
        return cls.ls()
    

        
