import commune as c
from typing import *
import os
from glob import glob

class Tree(c.Module):
    base_module = c.base_module # 

    @classmethod
    def tree(cls, search=None, 
                mode='path', 
                update:bool = False,
                path = 'local_module_tree',
                **kwargs) -> List[str]:
        
        module_tree = None
        if not update:
            if cls.module_tree_cache != None:
                return cls.module_tree_cache
            module_tree = c.get(path, None)
        if module_tree == None:

            assert mode in ['path', 'object']
            module_tree = {}

            # get the python paths
            python_paths = c.get_module_python_paths()
            module_tree ={c.path2simple(f): f for f in python_paths}

            if mode == 'object':
                module_tree = {f:c.path2objectpath(f) for f in module_tree.values()}

            # to use functions like c. we need to replace it with module lol
            if cls.root_module_class in module_tree:
                module_tree[cls.module_path()] = module_tree.pop(cls.root_module_class)
            
            c.put(path, module_tree)

        
        if search != None:
            module_tree = {k:v for k,v in module_tree.items() if search in k}

        return module_tree
    
    tree_folders_path = 'module_tree_folders'
    @classmethod
    def add_tree(cls, tree_path:str, **kwargs):
        path = cls.tree_folders_path
        tree_folder = c.get(path, [])
        tree_folder += [tree_path]
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
