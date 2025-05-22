
from .store import Store
import commune as c
import time

class TestStore:
    def __init__(self, module='~/.commune/store/test', **kwargs):
        self.store = Store(folder=module, **kwargs)

    def test_encrypt(self, path: str= 'test/a',  key: str=None) -> str:
            """
            Test the encryption and decryption of a file
            """

            if self.store.exists(path):
                self.store.rm(path)
            assert not self.store.exists(path), f'Failed to delete {path}'
            
            value = {'test': 'test', 'fam': {'test': 'test'}}
            self.store.put(path, value)
            obj = self.store.get(path)
            assert self.store.exists(path), f'Failed to find {path}'
            key = c.key(key)
            self.store.encrypt(path, key)
            assert self.store.is_encrypted(path), f'Failed to encrypt {path}'
            self.store.decrypt(path, key)
            assert not self.store.is_encrypted(path), f'Failed to decrypt {path}'
            # delete the file
            self.store.rm(path)
            assert not self.store.exists(path), f'Failed to delete {path}'
            return {'success': True, 'msg': 'Passed all tests'}


    def test_basics(self, path='test.json', data={'test': 'test', 'fam': {'test': 'test'}}):
        t0 = time.time()
        if self.store.exists(path):
            self.store.rm(path)
        assert not self.store.exists(path), f'Failed to delete'
        self.store.put(path, {'test': 'test'})
        assert self.store.exists(path), f'Failed to find {path}'
        data = self.store.get(path)
        self.store.rm(path)
        assert not self.store.exists(path), f'Failed to delete {path}'
        assert data == {'test': 'test'}, f'Failed test data={data}'
        t1 = time.time()
        print(f'Passed all tests in {t1 - t0} seconds')
        return {'success': True, 'msg': 'Passed all tests'}


    def test_encrypt_all(self, path2data={'test': 'test', 'fam': {'test': 'test'}}):
        for path, data in path2data.items():
            self.store.put(path, data)
        self.store.encrypt_all()
        assert all([self.store.is_encrypted(path) for path in path2data.keys()]), f'Failed to encrypt all {path2data.keys()}'
        self.store.decrypt_all()
        assert all([not self.store.is_encrypted(path) for path in path2data.keys()]), f'Failed to decrypt all {path2data.keys()}'
        for path in path2data.keys():
            self.store.rm(path)
        