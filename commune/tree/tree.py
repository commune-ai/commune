import commune as c
from typing import *
import os
from copy import deepcopy

class Tree(c.Module):
    tree_cache = {} # cache for tree

    def __init__(self, **kwargs):
        self.set_config(kwargs=locals())
        # c.thread(self.run_loop)
    

    @classmethod
    def simple2path(cls, path:str, tree = None, **kwargs) -> bool:
        pwd = c.pwd()
        simple_path = path
        dirpath = pwd + '/' + path.replace('.', '/')
        if os.path.isdir(dirpath):
            paths_in_dir = os.listdir(dirpath)
            for p in paths_in_dir:
                if p.endswith('.py'):
                    simple_path_filename = simple_path.replace('.', '_')
                    path_options = [
                                    simple_path_filename, 
                                    simple_path_filename + '_module', 
                                    'module_'+ simple_path_filename , 
                                    'main', 
                                    '__init__'
                                    ]
                    path_options = [f if f.endswith('.py') else f + '.py'  for f in path_options ]
                    for path_option in path_options:
                        if os.path.exists(path_option):
                            return path_option
        filepath = pwd + '/' + (path if path.endswith('.py') else path + '.py')
        if os.path.exists(filepath):
            full_path = filepath 
        else:
            tree = c.tree()
            is_module_in_tree = bool(simple_path in tree)
            if not is_module_in_tree:
                c.print(f'Path not found in tree: {simple_path}', color='red')
                tree = cls.tree(update=True, include_root=True)
            full_path = tree[simple_path]

        return full_path
    
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
    def default_tree_path(cls):
        return c.libpath
    
    
    @classmethod
    def tree(cls, 
                path = None,
                search=None,
                update = False,
                max_age = None, 
                include_root = False,
                verbose = True,
                cache = True,
                save = True,
                **kwargs
                ) -> List[str]:
        tree = {}
        mode = None
        timestamp = c.time()

        path = os.path.abspath(path or cls.default_tree_path())
        cache_path = path.split('/')[-1]
        # the tree is cached in memory to avoid repeated reads from storage
        # if the tree is in the cache and the max_age is not None, we want to check the age of the cache
        use_tree_cache = bool(cache and cache_path in cls.tree_cache) and not update
        if use_tree_cache:
            tree_data = cls.tree_cache[cache_path]
            assert all([k in tree_data for k in ['data', 'timestamp']]), 'Invalid tree cache'
            cache_age = timestamp - tree_data['timestamp']
            if max_age != None and cache_age < max_age:
                tree = tree_data['data']
                assert isinstance(tree, dict), 'Invalid tree data'
                mode = 'memory_cache'

        if len(tree) == 0:
            # if the tree is not cached in memory or we want to check the storage cache
            tree =  cls.get(cache_path, {}, max_age=max_age, update=update)
            mode = 'storage_cache'
            if len(tree) == 0 :
                # if the tree is not in the storage cache, we want to build it and store it
                mode = 'build'
                tree = cls.build_tree(path)
                cls.tree_cache[cache_path] = {'data': tree, 'timestamp': timestamp}
                cls.put(cache_path, tree)
        
        assert mode != None, 'Invalid mode'
        if search != None:
            tree = {k:v for k,v in tree.items() if search in k}
        if include_root:
            tree = {**tree, **cls.root_tree()}
        
        if verbose:
            latency = c.time() - timestamp
            c.print(f'Tree  path={path} latency={latency}, n={len(tree)} mode={mode}', color='cyan')
        return tree
    
    
    @classmethod
    def local_tree(cls, **kwargs):
        return cls.build_tree(c.pwd(), **kwargs)
    

    @classmethod
    def build_tree(cls,
                    tree_path:str = None, 
                    extension = '.py', 
                    verbose = True,
                    search=None,
                   **kwargs):
        
        tree_path = tree_path or c.libpath
        t1 = c.time()
        tree_path = cls.resolve_path(tree_path)
        module_tree = {}
        for path in c.glob(tree_path+'/**/**.py', recursive=True):
            simple_path = cls.path2simple(path)
            if simple_path == None:
                continue
            module_tree[simple_path] = path
        latency = c.time() - t1
        if search != None:
            module_tree = {k:v for k,v in module_tree.items() if search in k}
        c.print(f'Tree updated -> path={tree_path} latency={latency}, n={len(module_tree)}',  color='cyan', verbose=verbose)

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
    def path2simple(cls,  
                    path:str, 
                    tree = None,  
                    ignore_prefixes = ['src', 'commune', 'modules', 'router'],
                    module_folder_filnames = ['__init__', 'main', 'module'],
                    module_extension = 'py',
                    ignore_suffixes = ['module'],
                    compress_path = True,
                    verbose = False,
                    num_lines_to_read = 30,
                    ) -> str:
        
        
        
        path  = os.path.abspath(path)
        path_filename_with_extension = path.split('/')[-1] # get the filename with extension     
        path_extension = path_filename_with_extension.split('.')[-1] # get the extension
        assert path_extension == module_extension, f'Invalid extension {path_extension} for path {path}'
        path_filename = path_filename_with_extension[:-len(path_extension)-1] # remove the extension
        path_filename_chunks = path_filename.split('_')
        path_chunks = path.split('/')
        is_module_folder = all([bool(chunk in path_chunks) for chunk in path_filename_chunks])

        if path_filename in module_folder_filnames and not is_module_folder:
            initial_text = c.get_text(path, end_line=num_lines_to_read)
            if 'class ' in initial_text:
                is_module_folder = True
                c.print(f'Path {path} is a module folder', color='yellow', verbose=verbose)
            else:
                return None
            


        if is_module_folder:
            path = '/'.join(path.split('/')[:-1])

        # STEP 2 REMOVE THE TREE PATH (IF IT EXISTS)
        # if the path is a module folder, we want to remove the folder name
        tree_path = tree or c.libpath
        if path.startswith(tree_path):
            path = path[len(tree_path):]
        else:
            # if the tree path is not in the path, we want to remove the root path
            pwd = c.pwd()
            path = path[len(pwd):]  

        if path.startswith('/'):
            path = path[1:]
        


        # strip the extension
        path = path.replace('/', '.')

        # remove the extension
        module_extension = '.'+module_extension
        if path.endswith(module_extension):
            path = path[:-len(module_extension)]

        if compress_path:
            # we want to remove redundant chunks 
            # for example if the path is 'module/module' we want to remove the redundant module
            path_chunks = path.split('.')
            simple_path = []
            for chunk in path_chunks:
                if chunk not in simple_path:
                    simple_path += [chunk]
            simple_path = '.'.join(simple_path)
        else:
            simple_path = path
        # remove prefixes from commune
        for prefix in ignore_prefixes:
            prefix += '.'
            if simple_path.startswith(prefix) and simple_path != prefix:
                simple_path = simple_path[len(prefix):]
                c.print(f'Prefix {prefix} in path {simple_path}', color='yellow', verbose=verbose)

        for suffix in ignore_suffixes:
            suffix = '.' + suffix
            if simple_path.endswith(suffix) and simple_path != suffix:
                simple_path = simple_path[:-len(suffix)]
                c.print(f'Suffix {prefix} in path {simple_path}', color='yellow', verbose=verbose)



        # remove leading and trailing dots
        if simple_path.startswith('.'):
            simple_path = simple_path[1:]
        if simple_path.endswith('.'):
            simple_path = simple_path[:-1]

        return simple_path

    @classmethod
    def path_config_exists(cls, path:str,
                            config_files = ['config.yaml', 'config.yml'],
                              config_extensions=['.yaml', '.yml']) -> bool:
        '''
        Checks if the path exists
        '''
        config_files += [path.replace('.py', ext) for ext in config_extensions]
        dirpath = os.path.dirname(path)
        dir_files =  os.listdir(dirpath)
        if os.path.exists(dirpath) and any([[f.endswith(cf) for cf in config_files] for f in dir_files]):
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
    
    
if __name__ == '__main__':
    c.print(Tree.run())


    