
import json
import os
import time



class Store:

    def __init__(self, dirpath='~/.commune/server', mode='json'):
        self.dirpath = self.abspath(dirpath)
        self.mode = mode

    def put(self, path, data):
        path = self.get_path(path)
        dirpath = '/'.join(path.split('/')[:-1])
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f)
        return path

    def get(self, path, default=None, max_age=None, update=False):
        path = self.get_path(path)
        if not os.path.exists(path):
            return default
        with open(path, 'r') as f:
            data = json.load(f)
        if update:
            max_age = 0
        if max_age != None:
            if time.time() - os.path.getmtime(path) > max_age:
                data = default
        return data

    def get_path(self, path):
        if not path.startswith(self.dirpath):
            path = f'{self.dirpath}/{path}'
        if not path.startswith('/'):
            path  = f'{self.dirpath}/{path}'
            if self.mode != None:
                if not path.endswith(f'.{self.mode}'):
                    path = f'{path}.{self.mode}'
        return path

    def rm(self, path):
        path = self.get_path(path)
        assert os.path.exists(path), f'Failed to find path {path}'
        os.remove(path)
        return path

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

    def paths(self, search=None, avoid=None):
        import glob
        paths = glob.glob(f'{self.dirpath}/**/*', recursive=True)
        paths = [self.abspath(p) for p in paths if os.path.isfile(p)]
        if search != None:
            paths = [p for p in paths if search in p]
        if avoid != None:
            paths = [p for p in paths if avoid not in p]
        return paths
        

    def exists(self, path):
        path = self.get_path(path)
        return os.path.exists(path)
        

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
        n0 = self.n()
        if self.exists(path):
            self.rm(path)
        assert not self.exists(path), f'Failed to delete'
        self.put('test.json', {'test': 'test'})
        n1 = self.n()
        assert n1 == n0 + 1, f'Failed to add item n0={n0} n1={n1}'
        assert self.exists(path), f'Failed to find {path}'
        data = self.get(path)
        self.rm(path)
        n2 = self.n()
        assert n2 == n0, f'Failed to delete item n0={n0} n2={n2}'
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
        cid =  self.sha256(content)
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
    

    def sha256(self, content: str) -> str:
        import hashlib
        sha256_hash = hashlib.sha256()
        sha256_hash.update(content.encode('utf-8'))
        return sha256_hash.hexdigest()