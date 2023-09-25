import commune as c
import os
class ModuleTree(c.Module):

    @classmethod
    def resolve_tree_path(cls, path) -> str:
        assert c.isdir(path)
        return os.path.expanduser(path)

    @classmethod
    def add_tree(cls, tree, path):
        assert not c.isdir(path)
        trees = cls.get(tree, {'path': path, 'tree': {}})
        return cls.put('trees', trees )
    
    @classmethod
    def trees(cls):
        return cls.ls()

    @classmethod
    def get_tree_path(cls, tree):


    def exists()
    def rm_tree(cls, tree):
        assert c.isdir()
