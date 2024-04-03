import commune as c
from typing import *
import os
from copy import deepcopy

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
                path = 'local_module_tree',
                max_age = 60,
                trees = None
                ) -> List[str]:
        module_tree = {}

        t1 = c.time()

        trees = trees or cls.trees()
        if not hasattr(cls, 'tree_cache'):
            cls.tree_cache = {}


        module_tree =  c.get(path, {}, max_age=max_age)
        cls.tree_cache = module_tree
        

        if len(module_tree) == 0:
            for tree_path in cls.tree_paths():
                # get modules from each tree
                python_paths = c.get_module_python_paths(path=tree_path)
                # add the modules to the module tree
                new_tree = {c.path2simple(f, trees=trees): f for f in python_paths}
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
    def tree_paths(cls, update=False):
        path = cls.tree_folders_path
        trees =   [] if update else c.get(path, [])
        if len(trees) == 0:
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
                tree = None,
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
    
            tree2path = cls.tree2path()
            c.print(tree2path, 'tree2path')
            if tree != None:
                tree_paths = [tree]
            else:
                tree_paths = cls.tree_paths()
            for tree_path in tree_paths:
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
    def trees(cls, search=None):
        tree_paths = cls.tree_paths()
        trees = [t.split('/')[-1] for t in tree_paths]
        return trees

    @classmethod
    def tree2path(cls, name : str = None) -> str:
        tree_paths = cls.tree_paths()
        tree2path = {t.split('/')[-1]: t for t in tree_paths}
        if name != None:
            return tree2path[name]
        return tree2path


    @classmethod
    def path2simple(cls, path:str, trees=None) -> str:

        # does the config exist

        simple_path =  path.split(deepcopy(cls.root_dir))[-1]

        if cls.path_config_exists(path):
            simple_path = os.path.dirname(simple_path)

        simple_path = simple_path.replace('.py', '')
        
        simple_path = simple_path.replace('/', '.')[1:]

        # compress nae
        chunks = simple_path.split('.')
        simple_chunk = []
        for i, chunk in enumerate(chunks):
            if len(simple_chunk)>0:

                if simple_chunk[-1] == chunks[i]:
                    continue
                elif any([chunks[i].endswith(s) for s in ['_module', 'module']]):
                    continue
            simple_chunk += [chunk]
        
        if '_' in simple_chunk[-1]:
            filename_chunks = simple_chunk[-1].split('_')
            # if all of the chunks are in the filename
            if all([c in simple_chunk for c in filename_chunks]):
                simple_chunk = simple_chunk[:-1]

        simple_path = '.'.join(simple_chunk)

        # remove any files to compress the name even further for
        if len(simple_path.split('.')) > 2:
            
            if simple_path.split('.')[-1].endswith(simple_path.split('.')[-2]):
                simple_path = '.'.join(simple_path.split('.')[:-1])

        if trees != None:
            for tree in trees:
                if simple_path.startswith(tree):
                    simple_path = simple_path.replace(tree, '')
                    
        if simple_path.startswith('modules.'):
            simple_path = simple_path.replace('modules.', '')
        
        return simple_path
    