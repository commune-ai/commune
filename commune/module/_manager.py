
from typing import *
import os

class Manager:

    @classmethod
    def resolve_extension(cls, filename:str, extension = '.py') -> str:
        if filename.endswith(extension):
             return filename
        return filename + extension

    @classmethod
    def simple2path(cls, 
                    simple:str,
                    extension = '.py',
                    avoid_dirnames = ['', 'src', 
                                      'commune', 
                                      'commune/module', 
                                      'commune/modules', 
                                      'modules', 
                                      'blocks', 
                                      'agents', 
                                      'commune/agents'],
                    **kwargs) -> bool:
        """
        converts the module path to a file path

        for example 

        model.openai.gpt3 -> model/openai/gpt3.py, model/openai/gpt3_module.py, model/openai/__init__.py 
        model.openai -> model/openai.py or model/openai_module.py or model/__init__.py

        Parameters:
            path (str): The module path
        """
        # if cls.libname in simple and '/' not in simple and cls.can_import_module(simple):
        #     return simple
        shortcuts = cls.shortcuts()
        simple = shortcuts.get(simple, simple)

        if simple.endswith(extension):
            simple = simple[:-len(extension)]

        path = None
        pwd = cls.pwd()
        path_options = []
        simple = simple.replace('/', '.')

        # create all of the possible paths by combining the avoid_dirnames with the simple path
        dir_paths = list([pwd+ '/' + x for x in avoid_dirnames]) # local first
        dir_paths += list([cls.libpath + '/' + x for x in avoid_dirnames]) # add libpath stuff

        for dir_path in dir_paths:
            if dir_path.endswith('/'):
                dir_path = dir_path[:-1]
            # '/' count how many times the path has been split
            module_dirpath = dir_path + '/' + simple.replace('.', '/')
            if os.path.isdir(module_dirpath):
                simple_filename = simple.replace('.', '_')
                filename_options = [simple_filename, simple_filename + '_module', 'module_'+ simple_filename] + ['module'] + simple.split('.') + ['__init__']
                path_options +=  [module_dirpath + '/' + f  for f in filename_options]  
            else:
                module_filepath = dir_path + '/' + simple.replace('.', '/') 
                path_options += [module_filepath]
            for p in path_options:
                p = cls.resolve_extension(p)
                if os.path.exists(p):
                    p_text = cls.get_text(p)
                    path =  p
                    if 'commune' in p_text and 'class ' in p_text or '  def ' in p_text:
                        return p   
            if path != None:
                break
        return path

    
    @classmethod
    def is_repo(cls, libpath:str ):
        # has the .git folder
        return bool([f for f in cls.ls(libpath) if '.git' in f and os.path.isdir(f)])

    
    @classmethod
    def path2simple(cls,  
                    path:str, 
                    tree = None,  
                    ignore_prefixes = ['src', 'commune', 'modules', 'commune.modules',
                                       'commune.commune',
                                        'commune.module', 'module', 'router'],
                    module_folder_filnames = ['__init__', 'main', 'module'],
                    module_extension = 'py',
                    ignore_suffixes = ['module'],
                    name_map = {'commune': 'module'},
                    compress_path = True,
                    verbose = False,
                    num_lines_to_read = 100,
                    ) -> str:
        
        path  = os.path.abspath(path)
        path_filename_with_extension = path.split('/')[-1] # get the filename with extension     
        path_extension = path_filename_with_extension.split('.')[-1] # get the extension
        assert path_extension == module_extension, f'Invalid extension {path_extension} for path {path}'
        path_filename = path_filename_with_extension[:-len(path_extension)-1] # remove the extension
        path_filename_chunks = path_filename.split('_')
        path_chunks = path.split('/')

        if path.startswith(cls.libpath):
            path = path[len(cls.libpath):]
        else:
            # if the tree path is not in the path, we want to remove the root path
            pwd = cls.pwd()
            path = path[len(pwd):] 
        dir_chunks = path.split('/')[:-1] if '/' in path else []
        is_module_folder = all([bool(chunk in dir_chunks) for chunk in path_filename_chunks])
        is_module_folder = is_module_folder or (path_filename in module_folder_filnames)
        if is_module_folder:
            path = '/'.join(path.split('/')[:-1])
        path = path[1:] if path.startswith('/') else path
        path = path.replace('/', '.')
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
        # FILTER PREFIXES  
        for prefix in ignore_prefixes:
            prefix += '.'
            if simple_path.startswith(prefix) and simple_path != prefix:
                simple_path = simple_path[len(prefix):]
                cls.print(f'Prefix {prefix} in path {simple_path}', color='yellow', verbose=verbose)
        # FILTER SUFFIXES
        for suffix in ignore_suffixes:
            suffix = '.' + suffix
            if simple_path.endswith(suffix) and simple_path != suffix:
                simple_path = simple_path[:-len(suffix)]
                cls.print(f'Suffix {suffix} in path {simple_path}', color='yellow', verbose=verbose)

        # remove leading and trailing dots
        if simple_path.startswith('.'):
            simple_path = simple_path[1:]
        if simple_path.endswith('.'):
            simple_path = simple_path[:-1]
        simple_path = name_map.get(simple_path, simple_path)
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
    def resolve_cache_path(self, path):
        path = path.replace("/", "_")
        if path.startswith('_'):
            path = path[1:]
        path = f'cached_path/{path}'
        return path
    
    @classmethod
    def cached_paths(cls):
        return cls.ls('cached_paths')
    

    @classmethod
    def find_classes(cls, path='./',  working=False):

        path = os.path.abspath(path)
        if os.path.isdir(path):
            classes = []
            generator = cls.glob(path+'/**/**.py', recursive=True)
            for p in generator:
                if p.endswith('.py'):
                    p_classes =  cls.find_classes(p )
                    if working:
                        for class_path in p_classes:
                            try:
                                cls.import_object(class_path)
                                classes += [class_path]
                            except Exception as e:
                                r = cls.detailed_error(e)
                                r['class'] = class_path
                                cls.print(r, color='red')
                                continue
                    else:
                        classes += p_classes
                        
            return classes
        
        code = cls.get_text(path)
        classes = []
        file_path = cls.path2objectpath(path)
            
        for line in code.split('\n'):
            if all([s in line for s in ['class ', ':']]):
                new_class = line.split('class ')[-1].split('(')[0].strip()
                if new_class.endswith(':'):
                    new_class = new_class[:-1]
                if ' ' in new_class:
                    continue
                classes += [new_class]
        classes = [file_path + '.' + c for c in classes]

        libpath_objpath_prefix = cls.libpath.replace('/', '.')[1:] + '.'
        classes = [c.replace(libpath_objpath_prefix, '') for c in classes]
        return classes
    



    @classmethod
    def find_class2functions(cls, path,  working=False):

        path = os.path.abspath(path)
        if os.path.isdir(path):
            class2functions = {}
            for p in cls.glob(path+'/**/**.py', recursive=True):
                if p.endswith('.py'):
                    object_path = cls.path2objectpath(p)
                    response =  cls.find_class2functions(p )
                    for k,v in response.items():
                        class2functions[object_path+ '.' +k] = v
            return class2functions

        code = cls.get_text(path)
        classes = []
        class2functions = {}
        class_functions = []
        new_class = None
        for line in code.split('\n'):
            if all([s in line for s in ['class ', ':']]):
                new_class = line.split('class ')[-1].split('(')[0].strip()
                if new_class.endswith(':'):
                    new_class = new_class[:-1]
                if ' ' in new_class:
                    continue
                classes += [new_class]
                if len(class_functions) > 0:
                    class2functions[new_class] = cls.copy(class_functions)
                class_functions = []
            if all([s in line for s in ['   def', '(']]):
                fn = line.split(' def')[-1].split('(')[0].strip()
                class_functions += [fn]
        if new_class != None:
            class2functions[new_class] = class_functions

        return class2functions
    
    @classmethod
    def path2objectpath(cls, path:str, **kwargs) -> str:
        libpath = cls.libpath 
        path.replace
        if path.startswith(libpath):
            path =   path.replace(libpath , '')[1:].replace('/', '.').replace('.py', '')
        else: 
            pwd = cls.pwd()
            if path.startswith(pwd):
                path =  path.replace(pwd, '')[1:].replace('/', '.').replace('.py', '')
            
        return path.replace('__init__.', '.')
        

    @classmethod
    def find_functions(cls, path = './', working=False):
        fns = []
        if os.path.isdir(path):
            path = os.path.abspath(path)
            for p in cls.glob(path+'/**/**.py', recursive=True):
                p_fns = cls.find_functions(p)
                file_object_path = cls.path2objectpath(p)
                p_fns = [file_object_path + '.' + f for f in p_fns]
                for fn in p_fns:
                    if working:
                        try:
                            cls.import_object(fn)
                        except Exception as e:
                            r = cls.detailed_error(e)
                            r['fn'] = fn
                            cls.print(r, color='red')
                            continue
                    fns += [fn]

        else:
            code = cls.get_text(path)
            for line in code.split('\n'):
                if line.startswith('def ') or line.startswith('async def '):
                    fn = line.split('def ')[-1].split('(')[0].strip()
                    fns += [fn]
        return fns
    

    @classmethod
    def find_async_functions(cls, path):
        if os.path.isdir(path):
            path2classes = {}
            for p in cls.glob(path+'/**/**.py', recursive=True):
                path2classes[p] = cls.find_functions(p)
            return path2classes
        code = cls.get_text(path)
        fns = []
        for line in code.split('\n'):
            if line.startswith('async def '):
                fn = line.split('def ')[-1].split('(')[0].strip()
                fns += [fn]
        return [c for c in fns]
    
    @classmethod
    def find_objects(cls, path:str = './', search=None, working=False, **kwargs):
        classes = cls.find_classes(path, working=working)
        functions = cls.find_functions(path, working=working)

        if search != None:
            classes = [c for c in classes if search in c]
            functions = [f for f in functions if search in f]
        object_paths = functions + classes
        return object_paths
    objs = find_objects

    

    def find_working_objects(self, path:str = './', **kwargs):
        objects = self.find_objects(path, **kwargs)
        working_objects = []
        progress = self.tqdm(objects, desc='Progress')
        error_progress = self.tqdm(objects, desc='Errors')

        for obj in objects:

            try:
                self.import_object(obj)
                working_objects += [obj]
                progress.update(1)
            except:
                error_progress.update(1)
                pass
        return working_objects

    search = find_objects

    @classmethod
    def simple2objectpath(cls, 
                          simple_path:str,
                           cactch_exception = False, 
                           **kwargs) -> str:

        object_path = cls.simple2path(simple_path, **kwargs)
        classes =  cls.find_classes(object_path)
        return classes[-1]

    @classmethod
    def simple2object(cls, path:str, **kwargs) -> str:
        path =  cls.simple2objectpath(path, **kwargs)
        try:
            return cls.import_object(path)
        except:
            path = cls.tree().get(path)
            return cls.import_object(path)
            
    included_pwd_in_path = False
    @classmethod
    def import_module(cls, 
                      import_path:str, 
                      included_pwd_in_path=True, 
                      try_prefixes = ['commune','commune.modules', 'modules', 'commune.subspace', 'subspace']
                      ) -> 'Object':
        from importlib import import_module
        if included_pwd_in_path and not cls.included_pwd_in_path:
            import sys
            pwd = cls.pwd()
            sys.path.append(pwd)
            sys.path = list(set(sys.path))
            cls.included_pwd_in_path = True

        # if commune is in the path more than once, we want to remove the duplicates
        if cls.libname in import_path:
            import_path = cls.libname + import_path.split(cls.libname)[-1]
        pwd = cls.pwd()
        try:
            return import_module(import_path)
        except Exception as _e:
            for prefix in try_prefixes:
                try:
                    return import_module(f'{prefix}.{import_path}')
                except Exception as e:
                    pass
            raise _e
    
    @classmethod
    def can_import_module(cls, module:str) -> bool:
        '''
        Returns true if the module is valid
        '''
        try:
            cls.import_module(module)
            return True
        except:
            return False
    @classmethod
    def can_import_object(cls, module:str) -> bool:
        '''
        Returns true if the module is valid
        '''
        try:
            cls.import_object(module)
            return True
        except:
            return False

    @classmethod
    def import_object(cls, key:str, verbose: bool = 0, trials=3)-> Any:
        '''
        Import an object from a string with the format of {module_path}.{object}
        Examples: import_object("torch.nn"): imports nn from torch
        '''
        module = '.'.join(key.split('.')[:-1])
        object_name = key.split('.')[-1]
        if verbose:
            cls.print(f'Importing {object_name} from {module}')
        obj =  getattr(cls.import_module(module), object_name)
        return obj
    
    obj = get_obj = import_object


    @classmethod
    def object_exists(cls, path:str, verbose=False)-> Any:
        try:
            cls.import_object(path, verbose=verbose)
            return True
        except Exception as e:
            return False
    
    imp = get_object = importobj = import_object

    @classmethod
    def module_exists(cls, module:str, **kwargs) -> bool:
        '''
        Returns true if the module exists
        '''
        module_exists =  module in cls.modules(**kwargs)
        if not module_exists:
            try:
                module_path = cls.simple2path(module)
                module_exists = cls.exists(module_path)
            except:
                pass
        return module_exists
    
    @classmethod
    def has_app(cls, module:str, **kwargs) -> bool:
        return cls.module_exists(module + '.app', **kwargs)
    
    @classmethod
    def simplify_paths(cls,  paths):
        paths = [cls.simplify_path(p) for p in paths]
        paths = [p for p in paths if p]
        return paths

    @classmethod
    def simplify_path(cls, p, avoid_terms=['modules', 'agents']):
        chunks = p.split('.')
        if len(chunks) < 2:
            return None
        file_name = chunks[-2]
        chunks = chunks[:-1]
        path = ''
        for chunk in chunks:
            if chunk in path:
                continue
            path += chunk + '.'
        if file_name.endswith('_module'):
            path = '.'.join(path.split('.')[:-1])
        
        if path.startswith(cls.libname + '.'):
            path = path[len(cls.libname)+1:]

        if path.endswith('.'):
            path = path[:-1]

        if '_' in file_name:
            file_chunks =  file_name.split('_')
            if all([c in path for c in file_chunks]):
                path = '.'.join(path.split('.')[:-1])
        for avoid in avoid_terms:
            avoid = f'{avoid}.' 
            if avoid in path:
                path = path.replace(avoid, '')
        return path

    @classmethod
    def local_modules(cls, search=None):
        object_paths = cls.find_classes(cls.pwd())
        object_paths = cls.simplify_paths(object_paths) 
        if search != None:
            object_paths = [p for p in object_paths if search in p]
        return sorted(list(set(object_paths)))
    @classmethod
    def lib_tree(cls, ):
        return cls.get_tree(cls.libpath)
    @classmethod
    def local_tree(cls ):
        return cls.get_tree(cls.pwd())
    
    @classmethod
    def get_tree(cls, path):
        class_paths = cls.find_classes(path)
        simple_paths = cls.simplify_paths(class_paths) 
        return dict(zip(simple_paths, class_paths))
    
    @classmethod
    def get_module(cls, 
                   path:str = 'module',  
                   cache=True,
                   verbose = False,
                   update_tree_if_fail = True,
                   init_kwargs = None,
                   catch_error = False,
                   ) -> str:
        import commune as c
        path = path or 'module'
        if catch_error:
            try:
                return cls.get_module(path=path, cache=cache, 
                                      verbose=verbose, 
                                      update_tree_if_fail=update_tree_if_fail,
                                       init_kwargs=init_kwargs, 
                                       catch_error=False)
            except Exception as e:
                return c.detailed_error(e)
        if path in ['module', 'c']:
            return c.Module
        # if the module is a valid import path 
        shortcuts = c.shortcuts()
        if path in shortcuts:
            path = shortcuts[path]
        module = None
        cache_key = path
        t0 = c.time()
        if cache and cache_key in c.module_cache:
            module = c.module_cache[cache_key]
            return module
        module = c.simple2object(path)
        # ensure module
        if verbose:
            c.print(f'Loaded {path} in {c.time() - t0} seconds', color='green')
        
        if init_kwargs != None:
            module = module(**init_kwargs)
        is_module = c.is_module(module)
        if not is_module:
            module = cls.obj2module(module)
        if cache:
            c.module_cache[cache_key] = module            
        return module
    
    
    _tree = None
    @classmethod
    def tree(cls, search=None, cache=True):
        if cls._tree != None and cache:
            return cls._tree
        local_tree = cls.local_tree()
        lib_tree = cls.lib_tree()
        tree = {**local_tree, **lib_tree}
        if cache:
            cls._tree = tree
        if search != None:
            tree = {k:v for k,v in tree.items() if search in k}
        return tree

        return tree
    

    def overlapping_modules(self, search:str=None, **kwargs):
        local_modules = self.local_modules(search=search)
        lib_modules = self.lib_modules(search=search)
        return [m for m in local_modules if m in lib_modules]
    

    @classmethod
    def lib_modules(cls, search=None):
        object_paths = cls.find_classes(cls.libpath )
        object_paths = cls.simplify_paths(object_paths) 
        if search != None:
            object_paths = [p for p in object_paths if search in p]
        return sorted(list(set(object_paths)))
    
    @classmethod
    def find_modules(cls, search=None, **kwargs):
        local_modules = cls.local_modules(search=search)
        lib_modules = cls.lib_modules(search=search)
        return sorted(list(set(local_modules + lib_modules)))

    _modules = None
    @classmethod
    def modules(cls, search=None, cache=True,   **kwargs)-> List[str]:
        modules = cls._modules
        if not cache or modules == None:
            modules =  cls.find_modules(search=None, **kwargs)
        if search != None:
            modules = [m for m in modules if search in m]            
        return modules
    get_modules = modules

    @classmethod
    def has_module(cls, module):
        return module in cls.modules()
    



    
    def new_modules(self, *modules, **kwargs):
        for module in modules:
            self.new_module(module=module, **kwargs)



    @classmethod
    def new_module( cls,
                   module : str ,
                   base_module : str = 'demo', 
                   folder_module : bool = False,
                   update=1
                   ):
        
        import commune as c
        base_module = c.module(base_module)
        module_class_name = ''.join([m[0].capitalize() + m[1:] for m in module.split('.')])
        base_module_class_name = base_module.class_name()
        base_module_code = base_module.code().replace(base_module_class_name, module_class_name)
        pwd = c.pwd()
        path = os.path.join(pwd, module.replace('.', '/'))
        if folder_module:
            dirpath = path
            filename = module.replace('.', '_')
            path = os.path.join(path, filename)
        
        path = path + '.py'
        dirpath = os.path.dirname(path)
        if os.path.exists(path) and not update:
            return {'success': True, 'msg': f'Module {module} already exists', 'path': path}
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)

        c.put_text(path, base_module_code)
        
        return {'success': True, 'msg': f'Created module {module}', 'path': path}
    
    add_module = new_module


    @classmethod
    def has_local_module(cls, path=None):
        import commune as c 
        path = '.' if path == None else path
        if os.path.exists(f'{path}/module.py'):
            text = c.get_text(f'{path}/module.py')
            if 'class ' in text:
                return True
        return False
