
import json
import os
import time
import shutil
from typing import Optional, Union
import commune as c

class Store:

    def __init__(self, 
                folder='~/.commune/store', 
                suffix='json',
                key = None,
                private=False,
                ):

        """
        Store class to manage the storage of data in files

        folder: str: the path of the folder where the data is stored
        suffix: str: the suffix of the files (json, txt, etc)
        """
        self.folder = self.abspath(folder)
        self.key = self.get_key(key)
        self.suffix = suffix
        self.set_private(private) 

    def set_private(self, private: bool):
        if private:
            self.private = True
            self.encrypt_all()
        else:
            self.private = False
        
        return self.private
    
        
    def put(self, path, data, key=None):
        if self.private:
            key = self.key
        path = self.get_path(path, suffix=self.suffix)
        self.ensure_folder(path)
        with open(path, 'w') as f:
            json.dump(data, f)
        if key != None:
            key = self.get_key(key)
            self.encrypt(path, key=key)
        return {'path': path, 'encrypted': self.is_encrypted(path)}
    
    def shorten_item_path(self, path):
        return path.replace(self.folder+'/', '').replace(f'.{self.suffix}', '')

    def short2path(self):
        """
        Convert the short path to the full path
        """
        paths = self.paths()
        short2full = {}
        for p in paths:
            short = self.shorten_item_path(p)
            short2full[short] = p
        return short2full

    def ensure_folder(self, path):
        """
        Ensure that the directory exists
        """
        folder = os.path.dirname(path)
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        return {'path': path, 'folder': folder}

    def get(self, path, default=None, max_age=None, update=False, key=None):
        """
        Get the data from the file
        params
            path: str: the path of the file (relative to the self.folder)
            default: any: the default value to return if the file does not exist
            max_age: int: the maximum age of the file in seconds (update if too old)
            update: bool: if True, update the file if it is too old

        """
        path = self.get_path(path, suffix=self.suffix)
        if not os.path.exists(path):
            return default
        data = self.get_text(path)
        data = json.loads(data.strip())
        data = self.process_data_options(data)

        if not update:
            update =  bool(max_age != None and self.get_age(path) > max_age)
        if update:
            return default

        if key != None or self.private:
            key = self.get_key(key)
            try:
                data = key.decrypt(data["encrypted_data"])
            except Exception as e:
                print(f'Failed to decrypt {path} with key {key}. Error: {e}')
                return default
            return data

        return data

    def process_data_options(self, data):
    
        if isinstance(data, dict) and 'data' in data and ('time' in data or 'timestamp' in data):
            data = data['data']
        else: 
            return data

    def get_time(self, path, default=None):
        """
        Get the time of the file
        params
        """
        path = self.get_path(path, suffix=self.suffix)
        return os.path.getmtime(path)

    def get_age(self, path, default=None):
        """
        Get the age of the file
        params
        """
        path = self.get_path(path, suffix=self.suffix)
        if not os.path.exists(path):
            return default
        return time.time() - os.path.getmtime(path)

    

    def get_path(self, path:str, suffix:Optional[str]=None):
        """
        Get the path of the file
        params
            path: str: the path of the file
            suffix: str: the suffix of the file (json, txt, etc)
        return: str: the path of the file
        """
        if  path.startswith('~') or path.startswith('/') or path.startswith('./'):
            path = self.abspath(path)
        elif not path.startswith(self.folder):
            path = f'{self.folder}/{path}'
        if suffix != None:
            suffix = f'.{suffix}'
            if not path.endswith(suffix):
                path += suffix
        return path

    def in_folder(self, path):
        return path.startswith(self.folder)

    def rm(self, path):
        path = self.get_path(path, suffix=self.suffix)
        assert os.path.exists(path), f'Failed to find path {path}'
        assert self.in_folder(path), f'Path {path} is not in folder {self.folder}'
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        return path

    def rm_all(self):
        """
        Remove all items in the storage
        """
        paths = self.paths()
        for p in paths:
            self.rm(p)
        return paths

    def values(self, search=None, avoid=None, max_age=None):
        return [self.get(p) for p in self.paths(search=search, avoid=avoid, max_age=max_age)]
    
    def keys(self, search=None, avoid=None, max_age=None):
        """
        Get the keys in the storage
        """
        paths = self.paths(search=search, avoid=avoid, max_age=max_age)
        keys = [self.shorten_item_path(p) for p in paths]
        return keys

    def items(self, search=None,  key=None):
        """
        Get the items in the storage
        """
        keys = self.keys(search=search)
        data = []
        path2data = {}
        for p in keys:
            try:
                path2data[p] = self.get(p,key=key)
            except Exception as e:
                print(f'Failed to get {p} error={e}')
        return path2data
        

    def ls(self, path='./', search=None, avoid=None):
        path = self.get_path(path)
        if not os.path.exists(path):
            return []
        path = self.abspath(path)
        paths = os.listdir(path)
        paths = [f'{path}/{p}' for p in paths]
        return paths

    def lsdir(self, path='./', search=None, avoid=None):
        path = self.get_path(path)
        return os.listdir(path)

    def paths(self, path=None, search=None, avoid=None, max_age=None):
        import glob
        path = path or self.folder
        paths = glob.glob(f'{path}/**/*', recursive=True)
        paths = [self.abspath(p) for p in paths if os.path.isfile(p)]
        if search != None:
            paths = [p for p in paths if search in p]
        if avoid != None:
            paths = [p for p in paths if avoid not in p]
        if max_age != None:
            paths = [p for p in paths if time.time() - os.path.getmtime(p) < max_age]
        return paths

    def files(self, path=None, search=None, avoid=None):
        return self.paths(path=path,search=search, avoid=avoid)

        
    def exists(self, path):
        path = self.get_path(path)
        exists =  os.path.exists(path)
        if not exists:
            item_path = self.get_path(path, suffix=self.suffix)
            exists =  os.path.exists(item_path)
        return exists
    def item2age(self):
        """
        returns the age of the item in seconds
        """
        paths = self.paths()
        ages = {}
        for p in paths:
            ages[p] = time.time() - os.path.getmtime(p)
        return ages
        
    def n(self):
        paths = self.items()
        return len(paths)

    def _rm_all(self):
        """
        removes all items in the storage
        """
        paths = self.paths()
        for p in paths:
            os.remove(p)
        return paths


    def abspath(self, path):
        return os.path.abspath(os.path.expanduser(path))

    def path2age(self):
        """
        returns the age of the item in seconds
        """
        path2time = self.path2time()
        path2age = {}
        for p,t in path2time.items():
            path2age[p] = time.time() - t
        return path2age
    def path2time(self):
        """
        returns the time of the item in seconds
        """
        paths = self.paths()
        times = {}
        for p in paths:
            times[p] = os.path.getmtime(p)
        return times

    def cid(self, path, ignore_names=['__pycache__', '.DS_Store','.git', '.gitignore']):
        """
        Get the CID of the strat module
        """
        path = self.abspath(path)
        if os.path.isdir(path):
            files = os.listdir(path)
            content = []
            for f in files:
                if any([ignore in f for ignore in ignore_names]):
                    continue
                f = path + '/' + f
                content.append(self.cid(f))
            content = ''.join(content)
        elif os.path.isfile(path):
            content =  self.get_text(path)
        else: 
            raise Exception(f'Failed to find path {path}')
        cid =  self.hash(content)
        print(f'cid={cid} path={path}')
        return cid

    def get_text(self, path) -> str:
        with open(path, 'r') as f:
            result =  f.read()
        return result
    
    def hash(self, content: str, hash_type='sha256') -> str:
        import hashlib
        if hash_type == 'md5':
            hash_obj = hashlib.md5()
        elif hash_type == 'sha1':
            hash_obj = hashlib.sha1()
        elif hash_type == 'sha256':
            hash_obj = hashlib.sha256()
        else:
            raise ValueError(f'Unsupported hash mode: {mode}')
        hash_obj.update(content.encode('utf-8'))
        return hash_obj.hexdigest()

    def get_json(self, path: str= 'test/a')-> Union[dict, list]:
        path = self.get_path(path, suffix=self.suffix)
        data = self.get_text(path)
        data = json.loads(data)
        return data 

    def put_json(self, path: str= 'test/a', data: Union[dict, list]=None) -> str:
        json_data = json.dumps(data, indent=4)
        path = self.get_path(path, suffix=self.suffix)
        self.ensure_folder(path)
        with open(path, 'w') as f:
            f.write(json_data)
        return path

    def encrypt(self, path: str= 'test/a', key: str=None, password=None) -> str:
        """
        Encrypt a file using the given key
        """
        key = self.get_key(key)
        obj = self.get_json(path)
        assert self.exists(path), f'Failed to find {path}'
        if self.is_encrypted(path): 
            return {'msg': 'aready encrytped'}
        result = {'encrypted_data': key.encrypt(obj, password=password)}
        self.put(path,result)
        assert self.is_encrypted(path), f'Failed to encrypt {path}'
        return {'path': path, 'encrypted_data': result['encrypted_data']}

    def isdir(self, path: str= 'test') -> bool:
        """
        Check if the path is a directory
        """
        path = self.get_path(path)
        return os.path.isdir(path)

    def decrypt(self, path: str= 'test/a', key: str=None, password=None) -> str:
        """
        Decrypt a file using the given key
        """
        
        key = self.get_key(key)

        if isinstance(path, dict):
            return key.decrypt(path['encrypted_data'], password=password)    
        obj = self.get_json(path)
        if isinstance(obj, dict) and 'encrypted_data' in obj:
            result = key.decrypt(obj['encrypted_data'], password=password)
            self.put(path, result)
            assert not self.is_encrypted(path), f'Failed to decrypt {path}'
            return result
        else:
            raise Exception(f'Failed to decrypt {path}')


    def is_encrypted(self, path: str= 'test/a') -> bool:
        """
        Check if the file is encrypted using the given key
        """
        obj = self.get_json(path) if isinstance(path, str) else path
        return bool(isinstance(obj, dict) and 'encrypted_data' in obj)

    def is_private(self, path=None) -> bool:
        """
        Check if the file is private
        """
        return all([self.is_encrypted(p) for p in  self.paths(path=path)])

    def encrypted_paths(self, path=None) -> list:
        """
        Get the paths of the encrypted files
        """
        paths = self.paths(path=path)
        encrypted_paths = []
        for p in paths:
            if self.is_encrypted(p):
                encrypted_paths.append(p)
        return encrypted_paths
    
    def unencrypted_paths(self, path=None) -> list: 
        """
        Get the paths of the unencrypted files
        """
        paths = self.paths(path=path)
        unencrypted_paths = []
        for p in paths:
            if not self.is_encrypted(p):
                unencrypted_paths.append(p)
        return unencrypted_paths

    def encrypted(self) -> bool:
        encrypted_paths = self.encrypted_paths()
        # remove the folder from the path
        encrypted = [path.replace(self.folder+'/', '')for path in encrypted_paths]
        # remove the suffix from the path
        suffix = f'.{self.suffix}' 
        encrypted = [path[:-len(suffix)] if path.endswith(suffix) else path for path in encrypted]
        return encrypted

        return bool(isinstance(obj, dict) and 'encrypted_data' in obj and key.decryptable(obj['encrypted_data']))
    def path2name(self, path: str) -> str:
        if path.startswith(self.folder):
            path = path[len(self.folder)+1:]
        if path.endswith('.json'):
            path = path[:-len('.json')]
        return path

    def get_key(self, key: str=None) -> str:
        if key == None and hasattr(self, 'key'):
            key = self.key
        return c.key(key)

    def encrypt_all(self, key: str=None) -> list:
        """
        Encrypt all files in the given path
        """
        key = self.get_key(key)
        encrypted_paths = []
        for p in self.paths():
            if not self.is_encrypted(p):
                encrypted_paths.append(self.encrypt(p, key))
        assert all([self.is_encrypted(p) for p in encrypted_paths]), f'Failed to encrypt all paths {encrypted_paths}'
        return self.stats()


    def decrypt_all(self, key: str=None) -> list:
        """
        Decrypt all files in the given path
        """
        key = self.get_key(key)
        paths = self.paths()
        decrypted_paths = []
        for p in paths:
            if self.is_encrypted(p):
                decrypted_paths.append(self.decrypt(p, key))
        return self.stats()


    def stats(self, path = None)-> 'df':
        """
        Get the overview of the storage
        """
        path = self.get_path(path) if path else self.folder
        paths = self.paths(path)
        data = []
        for p in paths:
            data.append({'path': p.replace(path+'/', '')[:-len('.json')], 'age': self.get_age(p), 'size': os.path.getsize(p), 'encrypted': self.is_encrypted(p)})
        return c.df(data)
