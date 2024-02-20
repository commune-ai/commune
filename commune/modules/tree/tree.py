import commune as c
from typing import *
import os
from glob import glob

class Tree(c.Module):
    base_module = c.base_module # 

    @classmethod
    def add_tree(cls, tree, path):
        assert not c.isdir(path)
        trees = cls.get(tree, {'path': path, 'tree': {}})
        return cls.put('trees', trees )
    

    @classmethod
    def build_tree(cls, 
                update:bool = False,
                verbose:bool = False) -> List[str]:
                
        if update and verbose:
            c.print('Building module tree', verbose=verbose)
        module_tree = {cls.path2simple(f):f for f in cls.get_module_python_paths()}
        if cls.root_module_class in module_tree:
            module_tree['module'] = module_tree.pop(cls.root_module_class)
        return module_tree
    

    module_python_paths = None
    @classmethod
    def get_module_python_paths(cls) -> List[str]:
        '''
        Search for all of the modules with yaml files. Format of the file
        '''
        if isinstance(cls.module_python_paths, list): 
            return cls.module_python_paths
        modules = []

        # find all of the python files
        for f in glob(c.root_path + '/**/*.py', recursive=True):
            if os.path.isdir(f):
                continue
            file_path, file_ext =  os.path.splitext(f)
   
            if file_ext == '.py':
                dir_path, file_name = os.path.split(file_path)
                dir_name = os.path.basename(dir_path)

                if dir_name.lower() == file_name.lower():
                    # if the dirname is equal to the filename then it is a module
                    modules.append(f)
                if file_name.lower().endswith(dir_name.lower()):
                    # if the dirname is equal to the filename then it is a module
                    modules.append(f)
                if file_name.lower().endswith('module'):
                    # if the dirname is equal to the filename then it is a module
                    modules.append(f)
                    
                elif 'module' in file_name.lower():
                    modules.append(f)
                elif any([os.path.exists(file_path+'.'+ext) for ext in ['yaml', 'yml']]):
                    modules.append(f)
                else:
                    # FIX ME
                    f_classes = cls.find_python_class(f, search=['commune.Module', 'c.Module'])
                    # f_classes = []
                    if len(f_classes) > 0:
                        modules.append(f)

            
        cls.module_python_paths = modules
        
        return modules
    
    @classmethod
    def get_tree_root_dir(cls):
        tree_state = cls.get_tree_state()['path']


    @classmethod
    def path2simple(cls, path:str) -> str:

        # does the config exist

        simple_path =  path.split(c.copy(cls.root_dir))[-1]

        if cls.path_config_exists(path):
            simple_path = os.path.dirname(simple_path)

        simple_path = simple_path.replace('.py', '')
        
        
        simple_path = simple_path.replace('/', '.')[1:]

        # compress nae
        chunks = simple_path.split('.')
        new_chunks = []
        for i, chunk in enumerate(chunks):
            if len(new_chunks)>0:
                if new_chunks[-1] == chunks[i]:
                    continue
                elif any([chunks[i].endswith(s) for s in ['_module', 'module']]):
                    continue
            new_chunks.append(chunk)
        simple_path = '.'.join(new_chunks)
        
        # remove the modules prefix
        if simple_path.startswith('modules.'):
            simple_path = simple_path.replace('modules.', '')

        # remove any files to compress the name even further for
        if len(simple_path.split('.')) > 2:
            
            if simple_path.split('.')[-1].endswith(simple_path.split('.')[-2]):
                simple_path = '.'.join(simple_path.split('.')[:-1])
        return simple_path
    

Tree.run(__name__)