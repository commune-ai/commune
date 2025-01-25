
import commune as c

Key = c.module('key')
def test_signing( crypto_type=[1,2], data='test'):
    # at the moment, the ed25519 is not supported in the current version of pycryptodome
    if isinstance(crypto_type, list):
        return  [test_signing(k, data=data) for k in crypto_type]
    key = Key(crypto_type=crypto_type)
    sig = key.sign(data)
    assert key.verify(data,sig, key.public_key)
    return {'success':True, 'data':data, 'crypto_type':key.crypto_type2name(key.crypto_type)}


def test_encryption( values = [10, 'fam', 'hello world'], crypto_type=[0,1,2]):
    if isinstance(crypto_type, list):
        return [test_encryption(values=values, crypto_type=k) for k in crypto_type]
    for value in values:
        value = str(value)
        key = c.new_key(crypto_type=crypto_type)
        enc = key.encrypt(value)
        dec = key.decrypt(enc)
        assert dec == value, f'encryption failed, {dec} != {value}'
    return {'encrypted':enc, 'decrypted': dec, 'crypto_type':key.crypto_type2name(key.crypto_type)}
def test_encryption_with_password(value = 10, password = 'fam'):
    value = str(value)
    key = Key.new_key()
    enc = key.encrypt(value, password=password)
    dec = key.decrypt(enc, password=password)
    assert dec == value, f'encryption failed, {dec} != {value}'
    return {'encrypted':enc, 'decrypted': dec}

def test_key_encryption(test_key='test.key'):
    self = Key
    key = self.add_key(test_key, refresh=True)
    og_key = self.get_key(test_key)
    r = self.encrypt_key(test_key)
    self.decrypt_key(test_key, password=r['password'])
    key = self.get_key(test_key)
    assert key.ss58_address == og_key.ss58_address, f'key encryption failed, {key.ss58_address} != {self.ss58_address}'
    return {'success': True, 'msg': 'test_key_encryption passed'}

def test_key_management(key1='test.key' , key2='test2.key'):
    self = Key
    if self.key_exists(key1):
        self.rm_key(key1)
    if self.key_exists(key2):
        self.rm_key(key2)
    self.add_key(key1)
    k1 = self.get_key(key1)
    assert self.key_exists(key1), f'Key management failed, key still exists'
    self.mv_key(key1, key2)
    k2 = self.get_key(key2)
    assert k1.ss58_address == k2.ss58_address, f'Key management failed, {k1.ss58_address} != {k2.ss58_address}'
    assert self.key_exists(key2), f'Key management failed, key does not exist'
    assert not self.key_exists(key1), f'Key management failed, key still exists'
    self.mv_key(key2, key1)
    assert self.key_exists(key1), f'Key management failed, key does not exist'
    assert not self.key_exists(key2), f'Key management failed, key still exists'
    self.rm_key(key1)
    # self.rm_key(key2)
    assert not self.key_exists(key1), f'Key management failed, key still exists'
    assert not self.key_exists(key2), f'Key management failed, key still exists'
    return {'success': True, 'msg': 'test_key_management passed'}


def test_signing():
    self = Key()
    sig = self.sign('test')
    assert self.verify('test',sig, self.public_key)
    return {'success':True}

def test_key_encryption(password='1234'):
    path = 'test.enc'
    Key.add_key('test.enc', refresh=True)
    assert Key.is_key_encrypted(path) == False, f'file {path} is encrypted'
    Key.encrypt_key(path, password=password)
    assert Key.is_key_encrypted(path) == True, f'file {path} is not encrypted'
    Key.decrypt_key(path, password=password)
    assert Key.is_key_encrypted(path) == False, f'file {path} is encrypted'
    Key.rm(path)
    print('file deleted', path, c.path_exists, 'fam')
    assert not c.path_exists(path), f'file {path} not deleted'
    return {'success': True, 'msg': 'test_key_encryption passed'}

def test_move_key():
    self = Key()
    self.add_key('testfrom')
    assert self.key_exists('testfrom')
    og_key = self.get_key('testfrom')
    self.mv_key('testfrom', 'testto')
    assert self.key_exists('testto')
    assert not self.key_exists('testfrom')
    new_key = self.get_key('testto')
    assert og_key.ss58_address == new_key.ss58_address
    self.rm_key('testto')
    assert not self.key_exists('testto')
    return {'success':True, 'msg':'test_move_key passed', 'key':new_key.ss58_address}




