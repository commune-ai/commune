import commune as c
from typing import *
import os
from copy import deepcopy

class Tree(c.Module):
    _tree_cache = {}
    
    def __init__(self, **kwargs):
        self.set_config(kwargs=locals())
        # c.thread(self.run_loop)
    

    @classmethod
    def simple2path(cls, path:str, tree = None, update=False,     ignore_prefixes = ['commune', 'modules', 'commune.modules'], **kwargs) -> bool:
        pwd = c.pwd()
        simple_path = path
        path = c.pwd() + '/' + path.replace('.', '/')
        if os.path.isdir(path):
            paths_in_dir = os.listdir(path)
            for p in paths_in_dir:
                if p.endswith('.py'):
                    filename = p.split('.')[0].split('/')[-1]
                    if filename == simple_path:
                        path =  path +'/'+ p
                        break

        if os.path.exists(path + '.py'):
            path =  path + '.py'
        else:
            root_tree = cls.root_tree()
            tree = cls.tree()
            tree.update(root_tree)
            is_module_in_tree = bool(simple_path in tree)
            if not is_module_in_tree:
                tree = cls.tree(update=True, include_root=True)
                tree.update(root_tree)

            path = tree[simple_path]
            
        return path
    
    def path2tree(self, **kwargs) -> str:
        trees = c.trees()
        path2tree = {}
        for tree in trees:
            for module, path in self.tree(tree).items():
                path2tree[path] = tree
        return path2tree
    
    @classmethod
    def root_tree(cls, **kwargs):
        return cls.tree(path = c.libpath, include_root=False, **kwargs)

    @classmethod
    def is_repo(cls, libpath:str ):
        # has the .git folder
        return bool([f for f in cls.ls(libpath) if '.git' in f and os.path.isdir(f)])
    
    @classmethod
    def tree(cls, 
                path = None,
                search=None,
                update = False,
                max_age = None, 
                include_root = False,
                **kwargs
                ) -> List[str]:
        path = cls.resolve_path(path or c.pwd())
        is_repo = cls.is_repo(path)
        if not is_repo:
            path = c.libpath
        cache_path = path.split('/')[-1]
        if cache_path in cls._tree_cache:
            tree = cls._tree_cache[cache_path]
        else:
            tree =  c.get(cache_path, {}, max_age=max_age, update=update)
        if len(tree) == 0:
            tree = cls.build_tree(path)
            cls._tree_cache[cache_path] = tree
            cls.put(cache_path, tree)
        if search != None:
            tree = {k:v for k,v in tree.items() if search in k}
        if include_root:
            tree = {**tree, **cls.root_tree()}
        return tree
    
    @classmethod
    def local_tree(cls, **kwargs):
        return cls.build_tree(c.pwd(), **kwargs)
    

    @classmethod
    def build_tree(cls, tree_path:str = './', **kwargs):
        tree_path = cls.resolve_path(tree_path)
        module_tree = {}
        for root, dirs, files in os.walk(tree_path):
            for file in files:
                if file.endswith('.py') and '__init__' not in file and not file.split('/')[-1].startswith('_'):
                    path = os.path.join(root, file)
                    simple_path = cls.path2simple(path)
                    module_tree[simple_path] = path

        return module_tree
    @classmethod
    def tree_paths(cls, update=False, **kwargs) -> List[str]:
        return cls.ls()
    
    @classmethod
    def tree_hash(cls, *args, **kwargs):
        tree = cls.tree(*args, **kwargs)
        tree_hash = c.hash(tree)
        cls.put('tree_hash', tree_hash)
        return tree_hash

    @classmethod
    def old_tree_hash(cls, *args, **kwargs):
        return cls.get('tree_hash', None)

    @classmethod
    def has_tree_changed(cls, *args, **kwargs):
        old_tree_hash = cls.old_tree_hash(*args, **kwargs)
        new_tree_hash = cls.tree_hash(*args, **kwargs)
        return old_tree_hash != new_tree_hash

    def run_loop(self, *args, sleep_time=10, **kwargs):
        while True:
            c.print('Checking for tree changes')
            if self.has_tree_changed():
                self.tree(update=True)
            self.sleep(10)
        
    @classmethod
    def add_tree(cls, tree_path:str = './', **kwargs):
        return cls.tree(tree_path, update=True)
    
    @classmethod
    def rm_tree(cls, tree_path:str, **kwargs):
        return cls.rm(tree_path)
    
    @classmethod
    def rm_trees(cls, **kwargs):
        for tree_path in cls.tree_paths():
            cls.rm(tree_path)

    

    @classmethod
    def pwd_tree(cls):
        tree2path   =  c.tree2path()
        pwd = c.pwd()
        return {v:k for k,v in tree2path.items()}.get(pwd, None)

    @classmethod
    def is_pwd_tree(cls):
        return c.pwd() == c.libpath
    
    @classmethod
    def trees(cls):
        tree_paths = cls.tree_paths()
        trees = [t.split('/')[-1] for t in tree_paths]
        return trees

    @classmethod
    def tree2path(cls, tree : str = None, **kwargs) -> str:
        tree_paths = cls.tree_paths(**kwargs)
        tree2path = {t.split('/')[-1]: t for t in tree_paths}
        if tree != None:
            return tree2path[tree]
        return tree2path
    

    @classmethod
    def resolve_tree(cls, tree:str=None):
        if tree == None:    
            tree = root_tree_path.split('/')[-1]
        return tree
    
    

    @classmethod
    def path2simple(cls, path:str, ignore_prefixes = ['commune', 'modules', 'commune.modules']) -> str:

        path = os.path.abspath(path)
        pwd = c.pwd()
        if path.startswith(c.libpath):
            path = path.replace(c.libpath, '')
        elif path.startswith(pwd):
            path = path.replace(pwd, '')
        if cls.path_config_exists(path):
            simple_path = os.path.dirname(simple_path)
        else:
            simple_path = path
        simple_path = simple_path.replace('.py', '')
        simple_path = simple_path.replace('/', '.')
        if simple_path.startswith('.'):
            simple_path = simple_path[1:]
        # compress nae
        chunks = simple_path.split('.')
        simple_chunks = []
        simple_path = ''
        for i, chunk in enumerate(chunks):
            if len(simple_chunks)>0:
                if chunk in simple_chunks:
                    continue
            simple_chunks += [chunk]
            simple_path = '.'.join(simple_chunks)

        # FOR DIRECTORY MODULES: remove suffixes (e.g. _module, module, etc. or )
        suffix =  simple_path.split('.')[-1]
        if '_' in suffix:
            suffix = simple_path.split('.')[-1]
            suffix_chunks = suffix.split('_')
            new_simple_path = '.'.join(simple_path.split('.')[:-1])
            if all([s.lower() in new_simple_path for s in suffix_chunks]):
                simple_path = '.'.join(simple_path.split('.')[:-1])
        if suffix.endswith('_module'):
            simple_path = '.'.join(simple_path.split('.')[:-1])
        # remove prefixes from commune
        for prefix in ignore_prefixes:
            if simple_path.startswith(prefix):
                simple_path = simple_path.replace(prefix, '')

        # remove leading and trailing dots
        if simple_path.startswith('.'):
            simple_path = simple_path[1:]
        if simple_path.endswith('.'):
            simple_path = simple_path[:-1]
        
        return simple_path

    @classmethod
    def path_config_exists(cls, path:str) -> bool:
        '''
        Checks if the path exists
        '''
        for ext in ['.yaml', '.yml']:
            if os.path.exists(path.replace('.py', ext)):
                return True
        return False
    

    @classmethod
    def find_classes(cls, path):
        code = c.get_text(path)
        classes = []
        for line in code.split('\n'):
            if all([s in line for s in ['class ', ':']]):
                new_class = line.split('class ')[-1].split('(')[0].strip()
                if new_class.endswith(':'):
                    new_class = new_class[:-1]
                if ' ' in new_class:
                    continue
                classes += [new_class]
        return [c for c in classes]
    
    @classmethod
    def simple2objectpath(cls, simple_path:str, **kwargs) -> str:
        pwd = c.pwd()
        try:
            object_path = cls.simple2path(simple_path, **kwargs)
            classes =  cls.find_classes(object_path)
            if object_path.startswith(c.libpath):
                object_path = object_path[len(c.libpath):]
            object_path = object_path.replace('.py', '')
            if object_path.startswith(pwd):
                object_path = object_path[len(pwd):]

            object_path = object_path.replace('/', '.')
            if object_path.startswith('.'):
                object_path = object_path[1:]
            object_path = object_path + '.' + classes[-1]
        except Exception as e:
            object_path = simple_path
        return object_path
    
    




    