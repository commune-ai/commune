
import json
import os
import time

class Storage:

    def __init__(self, storage_dirpath='~/.storage', mode='json'):
        self.storage_dirpath = self.abspath(storage_dirpath)
        self.mode = mode

    def put(self, path, data):
        path = self.get_item_path(path)
        dirpath = '/'.join(path.split('/')[:-1])
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f)
        return path

    def get(self, path, default=None, max_age=None, update=False):
        path = self.get_item_path(path)
        with open(path, 'r') as f:
            data = json.load(f)
        if update:
            max_age = 0
        if max_age != None:
            if time.time() - os.path.getmtime(path) > max_age:
                data = default
        return data

    def get_item_path(self, path):
        if not path.startswith('/'):
            path  = f'{self.storage_dirpath}/{path}'
            if self.mode != None:
                if not path.endswith(f'.{self.mode}'):
                    path = f'{path}.{self.mode}'
        return path

    def rm(self, path):
        path = self.get_item_path(path)
        assert os.path.exists(path), f'Failed to find path {path}'
        os.remove(path)
        return path

    def items(self, df=False, features=None):
        paths = self.paths()
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

    def paths(self):
        import glob
        paths = glob.glob(f'{self.storage_dirpath}/**/*', recursive=True)
        return [self.abspath(p) for p in paths if os.path.isfile(p)]

    def exists(self, path):
        path = self.get_item_path(path)
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