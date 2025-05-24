
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
                encrypted=False,
                ):

        """
        Store class to manage the storage of data in files

        folder: str: the path of the folder where the data is stored
        suffix: str: the suffix of the files (json, txt, etc)
        """
        self.folder = self.abspath(folder)
        self.key = self.get_key(key)
        self.suffix = suffix
        self.encrypted = encrypted
        if self.encrypted:
            self.encrypt_all()

    def put(self, path, data):
        path = self.get_path(path, suffix=self.suffix)
        self.ensure_directory(path)
        with open(path, 'w') as f:
            json.dump(data, f)
        return path

    def ensure_directory(self, path):
        """
        Ensure that the directory exists
        """
        folder = os.path.dirname(path)
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        return {'path': path, 'folder': folder}

    def get(self, path, default=None, max_age=None, update=False):
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
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f'Failed to load {path} error={e}')
            data = default
        if isinstance(data, dict) and 'data' in data and ('time' in data or 'timestamp' in data):
            data = data['data']

            
        if not update:
            update =  bool(max_age != None and self.get_age(path) > max_age)
        if update:
            data = default
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
        if not path.startswith(self.folder):
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

    def rmdir(self, path):
        path = self.get_path(path)
        assert os.path.exists(path), f'Failed to find path {path}'
        return shutil.rmtree(path)

    def items(self, search=None, df=False, features=None):
        paths = self.paths(search=search)
        data = []
        for p in paths:
            try:
                data.append(self.get(p))
            except Exception as e:
                print(f'Failed to get {p} error={e}')
        if df:
            import pandas as pd
            data = pd.DataFrame(data)
        return data
        

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

    def paths(self, search=None, avoid=None, max_age=None):
        import glob
        paths = glob.glob(f'{self.folder}/**/*', recursive=True)
        paths = [self.abspath(p) for p in paths if os.path.isfile(p)]
        if search != None:
            paths = [p for p in paths if search in p]
        if avoid != None:
            paths = [p for p in paths if avoid not in p]
        if max_age != None:
            paths = [p for p in paths if time.time() - os.path.getmtime(p) < max_age]
        return paths

    def files(self, path=None, search=None, avoid=None):
        return self.paths(search=search, avoid=avoid)

        
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


    def encrypt(self, path: str= 'test/a', key: str=None, password=None) -> str:
        """
        Encrypt a file using the given key
        """
        key = self.get_key(key)

        obj = self.get(path)
        assert self.exists(path), f'Failed to find {path}'
        assert not self.is_encrypted(path), f'already encrypted {path}'
        result =  {'encrypted_data': key.encrypt(obj, password=password)}
        self.put(path, result)
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
        obj = self.get(path)
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
        obj = self.get(path)
        if isinstance(obj, dict) and 'encrypted_data' in obj:
            return True
        return False

    def encrypted_paths(self, key: str=None) -> list:
        """
        Get the paths of the encrypted files
        """
        key = self.get_key(key)
        paths = self.paths()
        encrypted_paths = []
        for p in paths:
            if self.is_encrypted(p):
                encrypted_paths.append(p)
        return encrypted_paths
    def path2name(self, path: str) -> str:
        if path.startswith(self.folder):
            path = path[len(self.folder)+1:]
        if path.endswith('.json'):
            path = path[:-len('.json')]
        return path


    def encrypted(self, key: str=None) -> list:
        encrypted_paths = self.encrypted_paths(key=key)
        results = []
        for p in encrypted_paths:
            n = self.path2name(p)
            results.append(n)
        return results


        # reverse 
    def get_key(self, key: str=None) -> str:
        if key == None and hasattr(self, 'key'):
            key = self.key
        return c.fn('key/get_key')(key)

    def encrypt_all(self, key: str=None) -> list:
        """
        Encrypt all files in the given path
        """
        key = self.get_key(key)
        encrypted_paths = []
        for p in self.paths():
            if not self.is_encrypted(p):
                encrypted_paths.append(self.encrypt(p, key))
        return encrypted_paths


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
        return decrypted_paths


    def stats(self)-> 'df':
        """
        Get the overview of the storage
        """
        paths = self.paths()
        data = []
        print('folder -->', self.folder)
        for p in paths:
            data.append({'path': p.replace(self.folder+'/', '')[:-len('.json')], 'age': self.get_age(p), 'size': os.path.getsize(p), 'encrypted': self.is_encrypted(p)})
        return c.df(data)
