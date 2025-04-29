import commune as c

Key = c.module('key')

class Test:

    def test_signing(self,  crypto_type=[0,1,2], data='test'):
        # at the moment, the ed25519 is not supported in the current version of pycryptodome
        if isinstance(crypto_type, list):
            return  [test_signing(k, data=data) for k in crypto_type]

        key = Key()
        crypto_type = key.get_crypto_type(crypto_type)
        key = Key(crypto_type=crypto_type)
        sig = key.sign(data)
        assert key.verify(data,sig, key.public_key)
        return {'success':True, 'data':data, 'crypto_type' : key.crypto_type}

    def test_encryption(self,  values = [10, 'fam', 'hello world'], crypto_type=[0,1,2]):
        if isinstance(crypto_type, list):
            return [self.test_encryption(values=values, crypto_type=k) for k in crypto_type]

        key = Key()
        crypto_type = key.get_crypto_type(crypto_type)
        for value in values:
            value = str(value)
            key = c.new_key(crypto_type=crypto_type)
            enc = key.encrypt(value)
            dec = key.decrypt(enc)
            assert dec == value, f'encryption failed, {dec} != {value}'
        return {'encrypted':enc, 'decrypted': dec, 'crypto_type':key.crypto_type}

    def test_encryption_with_password(self, value = 10, password = 'fam', crypto_type=[0,1,2]):
        if isinstance(crypto_type, list):
            return [self.test_encryption_with_password(value=value, password=password, crypto_type=k) for k in crypto_type]
        key = Key()
        crypto_type = key.get_crypto_type(crypto_type)
        value = str(value)
        key = key.new_key(crypto_type=crypto_type)
        enc = key.encrypt(value, password=password)
        dec = key.decrypt(enc, password=password)
        assert dec == value, f'encryption failed, {dec} != {value}'
        return {'encrypted':enc, 'decrypted': dec, 'crypto_type': crypto_type}

    def test_key_encryption(self, test_key='test.key', crypto_type=[0,1,2]):
        if isinstance(crypto_type, list):
            return [self.test_key_encryption(test_key=test_key, crypto_type=k) for k in crypto_type]
        key = Key()
        crypto_type = key.get_crypto_type(crypto_type)
        key = key.add_key(test_key, refresh=True)
        og_key = key.get_key(test_key)
        r = key.encrypt_key(test_key)
        key.decrypt_key(test_key, password=r['password'])
        key = key.get_key(test_key)
        assert key.ss58_address == og_key.ss58_address, f'key encryption failed, {key.ss58_address} != {self.ss58_address}'
        return {'success': True, 'msg': 'test_key_encryption passed'}

    def test_key_management(self, key1='test.key' , key2='test2.key', crypto_type=[0,1,2]):

        if isinstance(crypto_type, list):
            return [self.test_key_management(key1=key1, key2=key2, crypto_type=k) for k in crypto_type]
        key = Key()
        crypto_type = key.get_crypto_type(crypto_type)
        if key.key_exists(key1):
            key.rm_key(key1)
        if key.key_exists(key2):
            key.rm_key(key2)
        key.add_key(key1)
        k1 = key.get_key(key1)
        assert key.key_exists(key1), f'Key management failed, key still exists'
        key.mv_key(key1, key2)
        k2 = key.get_key(key2)
        assert k1.ss58_address == k2.ss58_address, f'Key management failed, {k1.ss58_address} != {k2.ss58_address}'
        assert key.key_exists(key2), f'Key management failed, key does not exist'
        assert not key.key_exists(key1), f'Key management failed, key still exists'
        key.mv_key(key2, key1)
        assert key.key_exists(key1), f'Key management failed, key does not exist'
        assert not key.key_exists(key2), f'Key management failed, key still exists'
        key.rm_key(key1)
        # self.rm_key(key2)
        assert not key.key_exists(key1), f'Key management failed, key still exists'
        assert not key.key_exists(key2), f'Key management failed, key still exists'
        return {'success': True, 'msg': 'test_key_management passed'}


    def test_signing(self, crypto_type=[1,2], data='test'):
        # TODO: for some reason, the ed25519 is not supported in the current version of pycryptodome
        for k in crypto_type:
            key = Key(crypto_type=k)
            sig = key.sign(data)
            assert key.verify(data,sig, public_key=key.public_key)
        key = Key()
        sig = key.sign('test')
        assert key.verify('test',sig, public_key=key.public_key)
        return {'success':True}

    def test_key_encryption(self, path = 'test.enc', password='1234'):
        key = Key()
        if key.key_exists(path):
            key.rm_key(path)
        key.add_key(path, refresh=True)
        assert key.is_key_encrypted(path) == False, f'file {path} is encrypted'
        key.encrypt_key(path, password=password)
        assert key.is_key_encrypted(path) == True, f'file {path} is not encrypted'
        key.decrypt_key(path, password=password)
        assert key.is_key_encrypted(path) == False, f'file {path} is encrypted'
        key.rm_key(path)
        assert not key.key_exists(path), f'file {path} not deleted'
        assert not c.path_exists(path), f'file {path} not deleted'
        return {'success': True, 'msg': 'test_key_encryption passed'}

    def test_move_key(self):
        key = Key()
        key.add_key('testfrom')
        assert key.key_exists('testfrom')
        og_key = key.get_key('testfrom')
        key.mv_key('testfrom', 'testto')
        assert key.key_exists('testto')
        assert not key.key_exists('testfrom')
        new_key = key.get_key('testto')
        assert og_key.ss58_address == new_key.ss58_address
        key.rm_key('testto')
        assert not key.key_exists('testto')
        return {'success':True, 'msg':'test_move_key passed', 'key':new_key.ss58_address}