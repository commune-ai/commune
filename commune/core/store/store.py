
import json
import os
import time
import shutil
from typing import Optional, Union
import commune as c

class Store:
    def __init__(self, folder_path='~/.commune/test', mode='json'):
        self.folder_path = self.abspath(folder_path)
        self.mode = mode

    def put(self, path, data):
        path = self.get_path(path, mode=self.mode)
        folder_path = '/'.join(path.split('/')[:-1])
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f)
        return path

    def get(self, path, default=None, max_age=None, update=False):
        path = self.get_path(path, mode=self.mode)
        folder_path = os.path.dirname(path)
        if not os.path.exists(path):
            return default
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f'Failed to load {path} error={e}')
            data = default
        if update:
            max_age = 0
        if max_age != None:
            if time.time() - os.path.getmtime(path) > max_age:
                data = default
        return data

    def get_path(self, path:str, mode:Optional[str]=None):
        """
        Get the path of the file
        params
            path: str: the path of the file
            mode: str: the mode of the file (json, txt, etc)
        return: str: the path of the file
        """
        if not path.startswith(self.folder_path):
            path = f'{self.folder_path}/{path}'
        if mode != None:
            suffix = f'.{mode}'
            if not path.endswith(suffix):
                path += suffix
        return path

    def rm(self, path):
        path = self.get_path(path, mode=self.mode)
        assert os.path.exists(path), f'Failed to find path {path}'
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

    def ls(self, path=None, search=None, avoid=None):
        path = self.get_path(path)
        if not os.path.exists(path):
            return []
        path = self.abspath(path)
        paths = os.listdir(path)
        paths = [f'{path}/{p}' for p in paths]
        return paths

    def paths(self, search=None, avoid=None, max_age=None):
        import glob
        paths = glob.glob(f'{self.folder_path}/**/*', recursive=True)
        paths = [self.abspath(p) for p in paths if os.path.isfile(p)]
        if search != None:
            paths = [p for p in paths if search in p]
        if avoid != None:
            paths = [p for p in paths if avoid not in p]
        if max_age != None:
            paths = [p for p in paths if time.time() - os.path.getmtime(p) < max_age]
        return paths
        
    def exists(self, path):
        path = self.get_path(path)
        exists =  os.path.exists(path)
        if not exists:
            item_path = self.get_path(path, mode=self.mode)
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

    def test(self, path='test.json', data={'test': 'test', 'fam': {'test': 'test'}}):
        t0 = time.time()
        if self.exists(path):
            self.rm(path)
        assert not self.exists(path), f'Failed to delete'
        self.put(path, {'test': 'test'})
        assert self.exists(path), f'Failed to find {path}'
        data = self.get(path)
        self.rm(path)
        assert not self.exists(path), f'Failed to delete {path}'
        assert data == {'test': 'test'}, f'Failed test data={data}'
        t1 = time.time()
        print(f'Passed all tests in {t1 - t0} seconds')
        return {'success': True, 'msg': 'Passed all tests'}

    def abspath(self, path):
        return os.path.abspath(os.path.expanduser(path))

    def path2age(self):
        """
        returns the age of the item in seconds
        """
        paths = self.paths()
        ages = {}
        for p in paths:
            ages[p] = time.time() - os.path.getmtime(p)
        return ages

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


    def get_age(self, path):
        """
        Get the age of the file in seconds
        """
        path = self.abspath(path)
        if os.path.exists(path):
            return time.time() - os.path.getmtime(path)
        else:
            raise Exception(f'Failed to find path {path}')
    def get_text(self, path) -> str:
        with open(path, 'r') as f:
            result =  f.read()
        return result
    
    def hash(self, content: str, mode='sha256') -> str:
        import hashlib
        if mode == 'md5':
            hash_obj = hashlib.md5()
        elif mode == 'sha1':
            hash_obj = hashlib.sha1()
        elif mode == 'sha256':
            hash_obj = hashlib.sha256()
        else:
            raise ValueError(f'Unsupported hash mode: {mode}')
        hash_obj.update(content.encode('utf-8'))
        return hash_obj.hexdigest()


    def encrypt(self, path: str= 'test/a', key: str=None) -> str:
        """
        Encrypt a file using the given key
        """
        import commune as c
        key = c.key(key)
        obj = self.get(path)
        result =  {'encrypted_data': key.encrypt(obj)}
        self.put(path, result)
        assert self.is_encrypted(path), f'Failed to encrypt {path}'
        return

    def decrypt(self, path: str= 'test/a', key: str=None) -> str:
        """
        Decrypt a file using the given key
        """
        import commune as c
        key = c.key(key)
        obj = self.get(path)
        if isinstance(obj, dict) and 'encrypted_data' in obj:
            result = key.decrypt(obj['encrypted_data'])
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

    def encrypted_paths(self, path: str= 'test/a', key: str=None) -> list:
        """
        Get the paths of the encrypted files
        """
        
        key = c.key(key)
        paths = self.paths(search=path)
        encrypted_paths = []
        for p in paths:
            if self.is_encrypted(p, key):
                encrypted_paths.append(p)
        return encrypted_paths

    def test_encrypt(self, path: str= 'test/a',  key: str=None) -> str:
        """
        Test the encryption and decryption of a file
        """

        if self.exists(path):
            self.rm(path)
        assert not self.exists(path), f'Failed to delete {path}'
        
        value = {'test': 'test', 'fam': {'test': 'test'}}
        self.put(path, value)
        obj = self.get(path)
        assert self.exists(path), f'Failed to find {path}'
        key = c.key(key)
        self.encrypt(path, key)
        assert self.is_encrypted(path), f'Failed to encrypt {path}'
        self.decrypt(path, key)
        assert not self.is_encrypted(path), f'Failed to decrypt {path}'
        # delete the file
        self.rm(path)
        assert not self.exists(path), f'Failed to delete {path}'
        return {'success': True, 'msg': 'Passed all tests'}