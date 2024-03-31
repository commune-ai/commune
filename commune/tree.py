import commune as c
from typing import *
import os

class Tree(c.Module):
    tree_folders_path = 'module_tree_folders'

    def __init__(self, a=1, b=2):
        self.set_config(kwargs=locals())

    def call(self, x:int = 1, y:int = 2) -> int:
        c.print(self.config)
        c.print(self.config, 'This is the config, it is a Munch object')
        return x + y
    

    @classmethod
    def tree(cls, search=None, 
                update:bool = False,
                verbose:bool = False,
                path = 'local_module_tree'
                ) -> List[str]:
        module_tree = {}

        t1 = c.time()

        if not hasattr(cls, 'tree_cache'):
            cls.tree_cache = {}

        if not update:
            if cls.tree_cache != {}:
                module_tree = cls.tree_cache
            else:
                module_tree =  c.get(path, {})
                cls.tree_cache = module_tree
        
        if len(module_tree) == 0:
            for tree_path in cls.trees():
                # get modules from each tree
                python_paths = c.get_module_python_paths(path=tree_path)
                # add the modules to the module tree
                new_tree = {c.path2simple(f): f for f in python_paths}
                for k,v in new_tree.items():
                    if k not in module_tree:
                        module_tree[k] = v
                # to use functions like c. we need to replace it with module lol
                if cls.root_module_class in module_tree:
                    module_tree[cls.root_module_class] = module_tree.pop(cls.root_module_class)
                
                c.put(path, module_tree)

        # cache the module tree
        if search != None:
            module_tree = {k:v for k,v in module_tree.items() if search in k}

        latency = c.time() - t1
        c.print(f'Loaded module tree in {latency} seconds', 
                color='green', 
                verbose=verbose)
        
        return module_tree
    
    @classmethod
    def trees(cls):
        path = cls.tree_folders_path
        trees =   c.get(path, [])
        if c.libpath not in trees:
            trees = cls.default_trees()
        return trees
    
        
    @classmethod
    def add_tree(cls, tree_path:str, **kwargs):

        tree_path = os.path.expanduser(tree_path)
        path = cls.tree_folders_path
        tree_folder = c.get(path, [])

        tree_folder += [tree_path]
        tree_folder = list(set(tree_folder))
        assert os.path.isdir(tree_path)
        assert isinstance(tree_folder, list)
        c.put(path, tree_folder, **kwargs)
        return {'module_tree_folders': tree_folder}
    
    @classmethod
    def rm_tree(cls, tree_path:str, **kwargs):
        path = cls.tree_folders_path
        tree_folder = c.get(tree_path, [])
        tree_folder = [f for f in tree_folder if f != tree_path ]
        c.put(path, tree_folder)
        return {'module_tree_folders': tree_folder}

    @classmethod
    def default_trees(cls):
        return [c.libpath + '/commune' ,
                 c.libpath + '/modules',
                  c.libpath + '/my_modules'
                   ]
    

    @classmethod
    def tree(cls, search=None, 
                update:bool = False,
                verbose:bool = False,
                path = 'local_module_tree'
                ) -> List[str]:
        module_tree = {}

        t1 = c.time()

        if not hasattr(cls, 'tree_cache'):
            cls.tree_cache = {}

        if not update:
            if cls.tree_cache != {}:
                module_tree = cls.tree_cache
            else:
                module_tree =  c.get(path, {})
                cls.tree_cache = module_tree
        
        if len(module_tree) == 0:
            for tree_path in cls.trees():
                # get modules from each tree
                python_paths = c.get_module_python_paths(path=tree_path)
                # add the modules to the module tree
                new_tree = {c.path2simple(f): f for f in python_paths}
                for k,v in new_tree.items():
                    if k not in module_tree:
                        module_tree[k] = v
                # to use functions like c. we need to replace it with module lol
                if cls.root_module_class in module_tree:
                    module_tree[cls.root_module_class] = module_tree.pop(cls.root_module_class)
                
                c.put(path, module_tree)

        # cache the module tree
        if search != None:
            module_tree = {k:v for k,v in module_tree.items() if search in k}

        latency = c.time() - t1
        c.print(f'Loaded module tree in {latency} seconds', 
                color='green', 
                verbose=verbose)
        
        return module_tree
    

    @classmethod
    def name2tree(cls, name : str = None) -> str:
        trees = cls.trees()
        name2tree = {t.split('/')[-1]: t for t in trees}
        if name != None:
            return name2tree[name]
        return name2tree
