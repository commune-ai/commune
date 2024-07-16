
from typing import *
import os
import glob
import inspect
from munch  import Munch
from copy import deepcopy
import yaml
import json
import time
import shutil

class Storage:

    @classmethod
    async def async_put_json(cls,*args,**kwargs) -> str:
        return cls.put_json(*args, **kwargs) 
    
    @classmethod
    def put_json(cls, 
                 path:str, 
                 data:Dict, 
                 meta = None,
                 verbose: bool = False,

                 **kwargs) -> str:
        if meta != None:
            data = {'data':data, 'meta':meta}
        path = cls.resolve_path(path=path, extension='json')
        # cls.lock_file(path)
        if isinstance(data, dict):
            data = json.dumps(data)
        cls.put_text(path, data)
        return path
    
    save_json = put_json
    


    @classmethod
    def rm_json(cls, path=None):
        from .utils.dict import rm_json

        if path in ['all', '**']:
            return [cls.rm_json(f) for f in cls.glob(files_only=False)]
        
        path = cls.resolve_path(path=path, extension='json')

        return rm_json(path )
    

    @classmethod
    def rmdir(cls, path):
        return shutil.rmtree(path)


    @classmethod
    def isdir(cls, path):
        path = cls.resolve_path(path=path)
        return os.path.isdir(path)
        
    
    @classmethod
    def isfile(cls, path):
        path = cls.resolve_path(path=path)
        return os.path.isfile(path)
    
    @classmethod
    def rm_all(cls):
        for path in cls.ls():
            cls.rm(path)
        return {'success':True, 'message':f'{cls.storage_dir()} removed'}
        


    @classmethod
    def rm(cls, path, extension=None, mode = 'json'):
        
        assert isinstance(path, str), f'path must be a string, got {type(path)}'
        path = cls.resolve_path(path=path, extension=extension)

        # incase we want to remove the json file
        mode_suffix = f'.{mode}'
        if not os.path.exists(path) and os.path.exists(path+mode_suffix):
            path += mode_suffix

        if not os.path.exists(path):
            return {'success':False, 'message':f'{path} does not exist'}
        if os.path.isdir(path):
            cls.rmdir(path)
        if os.path.isfile(path):
            os.remove(path)
        assert not os.path.exists(path), f'{path} was not removed'

        return {'success':True, 'message':f'{path} removed'}
    
    @classmethod
    def rm_all(cls):
        storage_dir = cls.storage_dir()
        if cls.exists(storage_dir):
            cls.rm(storage_dir)
        assert not cls.exists(storage_dir), f'{storage_dir} was not removed'
        cls.makedirs(storage_dir)
        assert cls.is_dir_empty(storage_dir), f'{storage_dir} was not removed'
        return {'success':True, 'message':f'{storage_dir} removed'}



    @classmethod
    def rm_all(cls):
        storage_dir = cls.storage_dir()
        if cls.exists(storage_dir):
            cls.rm(storage_dir)
        assert not cls.exists(storage_dir), f'{storage_dir} was not removed'
        cls.makedirs(storage_dir)
        assert cls.is_dir_empty(storage_dir), f'{storage_dir} was not removed'
        return {'success':True, 'message':f'{storage_dir} removed'}


    @classmethod
    def glob(cls,  path =None, files_only:bool = True, recursive:bool=True):
        path = cls.resolve_path(path, extension=None)
        if os.path.isdir(path):
            path = os.path.join(path, '**')
        paths = glob.glob(path, recursive=recursive)
        if files_only:
            paths =  list(filter(lambda f:os.path.isfile(f), paths))
        return paths
    

    @classmethod
    def put_cache(cls,k,v ):
        cls.cache[k] = v
    
    @classmethod
    def get_cache(cls,k, default=None, **kwargs):
        v = cls.cache.get(k, default)
        return v


    @classmethod
    def get_json(cls, 
                path:str,
                default:Any=None,
                verbose: bool = False,**kwargs):
        from .utils.dict import async_get_json
        path = cls.resolve_path(path=path, extension='json')

        cls.print(f'Loading json from {path}', verbose=verbose)

        try:
            data = cls.get_text(path, **kwargs)
        except Exception as e:
            return default
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception as e:
                return default
        if isinstance(data, dict):
            if 'data' in data and 'meta' in data:
                data = data['data']
        return data
    @classmethod
    async def async_get_json(cls,*args, **kwargs):
        return  cls.get_json(*args, **kwargs)

    load_json = get_json


    @classmethod
    def file_exists(cls, path:str)-> bool:
        path = cls.resolve_path(path)
        exists =  os.path.exists(path)
        return exists

 
    exists = exists_json = file_exists 


    @classmethod
    def filepath(cls, obj=None) -> str:
        '''
        removes the PWD with respect to where module.py is located
        '''
        obj = cls.resolve_object(obj)
        try:
            module_path =  inspect.getfile(obj)
        except Exception as e:
            print(f'Error: {e}')
            module_path =  inspect.getfile(cls)
        return module_path

    pythonpath = pypath =  filepath



    @classmethod
    def makedirs(cls, *args, **kwargs):
        return os.makedirs(*args, **kwargs)


    @classmethod
    def mv(cls, path1, path2):
        path1 = cls.resolve_path(path1)
        path2 = cls.resolve_path(path2)
        assert os.path.exists(path1), path1
        if not os.path.isdir(path2):
            path2_dirpath = os.path.dirname(path2)
            if not os.path.isdir(path2_dirpath):
                os.makedirs(path2_dirpath, exist_ok=True)
        shutil.move(path1, path2)
        assert os.path.exists(path2), path2
        assert not os.path.exists(path1), path1
        return path2

    @classmethod
    def resolve_path(cls, path:str = None, extension=None):
        '''
        ### Documentation for `resolve_path` class method
        
        #### Purpose:
        The `resolve_path` method is a class method designed to process and resolve file and directory paths based on various inputs and conditions. This method is useful for preparing file paths for operations such as reading, writing, and manipulation.
        
        #### Parameters:
        - `path` (str, optional): The initial path to be resolved. If not provided, a temporary directory path will be returned.
        - `extension` (Optional[str], optional): The file extension to append to the path if necessary. Defaults to None.
        - `root` (bool, optional): A flag to determine whether the path should be resolved in relation to the root directory. Defaults to False.
        - `file_type` (str, optional): The default file type/extension to append if the `path` does not exist but appending the file type results in a valid path. Defaults to 'json'.
        
        #### Behavior:
        - If `path` is not provided, the method returns a path to a temporary directory.
        - If `path` starts with '/', it is returned as is.
        - If `path` starts with '~/', it is expanded to the user’s home directory.
        - If `path` starts with './', it is resolved to an absolute path.
        - If `path` does not fall under the above conditions, it is treated as a relative path. If `root` is True, it is resolved relative to the root temp directory; otherwise, relative to the class's temp directory.
        - If `path` is a relative path and does not contain the temp directory, the method joins `path` with the appropriate temp directory.
        - If `path` does not exist as a directory and an `extension` is provided, the extension is appended to `path`.
        - If `path` does not exist but appending the `file_type` results in an existing path, the `file_type` is appended.
        - The parent directory of `path` is created if it does not exist, avoiding any errors when the path is accessed later.
        
        #### Returns:
        - `str`: The resolved and potentially created path, ensuring it is ready for further file operations. 
        
        #### Example Usage:
        ```python
        # Resolve a path in relation to the class's temporary directory
        file_path = MyClassName.resolve_path('data/subfolder/file', extension='txt')
        
        # Resolve a path in relation to the root temporary directory
        root_file_path = MyClassName.resolve_path('configs/settings'
        ```
        
        #### Notes:
        - This method relies on the `os` module to perform path manipulations and checks.
        - This method is versatile and can handle various input path formats, simplifying file path resolution in the class's context.
        '''
    
        if path == None:
            return cls.storage_dir()
        
        if path.startswith('/'):
            path = path
        elif path.startswith('~/'):
            path =  os.path.expanduser(path)
        elif path.startswith('.'):
            path = os.path.abspath(path)
        else:
            # if it is a relative path, then it is relative to the module path
            # ex: 'data' -> '.commune/path_module/data'
            storage_dir = cls.storage_dir()
            if storage_dir not in path:
                path = os.path.join(storage_dir, path)

        if extension != None and not path.endswith(extension):
            path = path + '.' + extension

        return path
    



    @classmethod
    def put_yaml(cls, path:str,  data: dict) -> Dict:
        '''
        Loads a yaml file
        '''
        path = cls.resolve_path(path)
            
        from .utils.dict import save_yaml
        if isinstance(data, Munch):
            data = cls.munch2dict(deepcopy(data))
            
        return save_yaml(data=data , path=path)
    

    @classmethod
    def get_yaml(cls, path:str=None, default={}, **kwargs) -> Dict:
        '''f
        Loads a yaml file
        '''
        path = cls.resolve_path(path)
        with open(path, 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)

        return data
    
        
    load_yaml = get_yaml

    save_yaml = put_yaml 
    
    @classmethod
    def filesize(cls, filepath:str):
        filepath = cls.resolve_path(filepath)
        return os.path.getsize(filepath)


    @classmethod
    def cp(cls, path1:str, path2:str, refresh:bool = False):
        # what if its a folder?
        assert os.path.exists(path1), path1
        if refresh == False:
            assert not os.path.exists(path2), path2
        
        path2_dirpath = os.path.dirname(path2)
        if not os.path.isdir(path2_dirpath):
            os.makedirs(path2_dirpath, exist_ok=True)
            assert os.path.isdir(path2_dirpath), f'Failed to create directory {path2_dirpath}'

        if os.path.isdir(path1):
            shutil.copytree(path1, path2)


        elif os.path.isfile(path1):
            
            shutil.copy(path1, path2)
        else:
            raise ValueError(f'path1 is not a file or a folder: {path1}')
        return {'success': True, 'msg': f'Copied {path1} to {path2}'}

    @classmethod
    def put_text(cls, path:str, text:str, key=None, bits_per_character=8) -> None:
        # Get the absolute path of the file
        path = cls.resolve_path(path)
        dirpath = os.path.dirname(path)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        if not isinstance(text, str):
            text = cls.python2str(text)
        if key != None:
            text = cls.get_key(key).encrypt(text)
        # Write the text to the file
        with open(path, 'w') as file:
            file.write(text)
        # get size
        text_size = len(text)*bits_per_character
    
        return {'success': True, 'msg': f'Wrote text to {path}', 'size': text_size}
    
    @classmethod
    def lsdir(cls, path:str) -> List[str]:
        path = os.path.abspath(path)
        return os.listdir(path)

    @classmethod
    def abspath(cls, path:str) -> str:
        return os.path.abspath(path)


    @classmethod
    def ls(cls, path:str = '', 
           recursive:bool = False,
           search = None,
           return_full_path:bool = True):
        """
        provides a list of files in the path 

        this path is relative to the module path if you dont specifcy ./ or ~/ or /
        which means its based on the module path
        """
        path = cls.resolve_path(path)
        try:
            ls_files = cls.lsdir(path) if not recursive else cls.walk(path)
        except FileNotFoundError:
            return []
        if return_full_path:
            ls_files = [os.path.abspath(os.path.join(path,f)) for f in ls_files]

        ls_files = sorted(ls_files)
        if search != None:
            ls_files = list(filter(lambda x: search in x, ls_files))
        return ls_files
    


    @classmethod
    def put(cls, 
            k: str, 
            v: Any,  
            mode: bool = 'json',
            encrypt: bool = False, 
            verbose: bool = False, 
            password: str = None, **kwargs) -> Any:
        '''
        Puts a value in the config
        '''
        encrypt = encrypt or password != None
        
        if encrypt or password != None:
            v = cls.encrypt(v, password=password)

        if not cls.jsonable(v):
            v = cls.serialize(v)    
        
        data = {'data': v, 'encrypted': encrypt, 'timestamp': cls.timestamp()}            
        
        # default json 
        getattr(cls,f'put_{mode}')(k, data)

        data_size = cls.sizeof(v)
    
        return {'k': k, 'data_size': data_size, 'encrypted': encrypt, 'timestamp': cls.timestamp()}
    
    @classmethod
    def get(cls,
            k:str, 
            default: Any=None, 
            mode:str = 'json',
            max_age:str = None,
            cache :bool = False,
            full :bool = False,
            key: 'Key' = None,
            update :bool = False,
            password : str = None,
            **kwargs) -> Any:
        
        '''
        Puts a value in sthe config, with the option to encrypt it

        Return the value
        '''
        if cache:
            if k in cls.cache:
                return cls.cache[k]
        data = getattr(cls, f'get_{mode}')(k,default=default, **kwargs)
        
            

        if password != None:
            assert data['encrypted'] , f'{k} is not encrypted'
            data['data'] = cls.decrypt(data['data'], password=password, key=key)

        data = data or default
        
        if isinstance(data, dict):
            if update:
                max_age = 0
            if max_age != None:
                timestamp = data.get('timestamp', None)
                if timestamp != None:
                    age = int(time.time() - timestamp)
                    if age > max_age: # if the age is greater than the max age
                        print(f'{k} is too old ({age} > {max_age})')
                        return default
        else:
            data = default
            
        if not full:
            if isinstance(data, dict):
                if 'data' in data:
                    data = data['data']

        # local cache
        if cache:
            cls.cache[k] = data
        return data
  


    
    @classmethod
    def get_text(cls, 
                 path: str, 
                 tail = None,
                 start_byte:int = 0,
                 end_byte:int = 0,
                 start_line :int= None,
                 end_line:int = None ) -> str:
        # Get the absolute path of the file
        path = cls.resolve_path(path)

        # Read the contents of the file
        with open(path, 'rb') as file:

            file.seek(0, 2) # this is done to get the fiel size
            file_size = file.tell()  # Get the file size
            if start_byte < 0:
                start_byte = file_size - start_byte
            if end_byte <= 0:
                end_byte = file_size - end_byte 
            if end_byte < start_byte:
                end_byte = start_byte + 100
            chunk_size = end_byte - start_byte + 1

            file.seek(start_byte)

            content_bytes = file.read(chunk_size)

            # Convert the bytes to a string
            try:
                content = content_bytes.decode()
            except UnicodeDecodeError as e:
                if hasattr(content_bytes, 'hex'):
                    content = content_bytes.hex()
                else:
                    raise e

            if tail != None:
                content = content.split('\n')
                content = '\n'.join(content[-tail:])
    
            elif start_line != None or end_line != None:
                
                content = content.split('\n')
                if end_line == None or end_line == 0 :
                    end_line = len(content) 
                if start_line == None:
                    start_line = 0
                if start_line < 0:
                    start_line = start_line + len(content)
                if end_line < 0 :
                    end_line = end_line + len(content)
                content = '\n'.join(content[start_line:end_line])
            else:
                content = content_bytes.decode()
        return content

    @classmethod
    def storage_dir(cls):
        return f'{cls.cache_path}/{cls.module_name()}'
    
    tmp_dir = cache_dir   = storage_dir
    
    @classmethod
    def refresh_storage(cls):
        cls.rm(cls.storage_dir())

    @classmethod
    def refresh_storage_dir(cls):
        cls.rm(cls.storage_dir())
        cls.makedirs(cls.storage_dir())
        

    @classmethod
    def rm_lines(cls, path:str, start_line:int, end_line:int) -> None:
        # Get the absolute path of the file
        text = cls.get_text(path)
        text = text.split('\n')
        text = text[:start_line-1] + text[end_line:]
        text = '\n'.join(text)
        cls.put_text(path, text)
        return {'success': True, 'msg': f'Removed lines {start_line} to {end_line} from {path}'}
    @classmethod
    def rm_line(cls, path:str, line:int, text=None) -> None:
        # Get the absolute path of the file
        text =  cls.get_text(path)
        text = text.split('\n')
        text = text[:line-1] + text[line:]
        text = '\n'.join(text)
        cls.put_text(path, text)
        return {'success': True, 'msg': f'Removed line {line} from {path}'}
        # Write the text to the file
            
    @classmethod
    def tilde_path(cls):
        return os.path.expanduser('~')

    def is_dir_empty(self, path:str):
        return len(self.ls(path)) == 0
    @classmethod
    def get_file_size(cls, path:str):
        path = cls.resolve_path(path)
        return os.path.getsize(path)

    @staticmethod
    def jsonable( value):
        import json
        try:
            json.dumps(value)
            return True
        except:
            return False