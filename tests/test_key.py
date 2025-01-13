
import commune as c
crypto_type='solana'
def test_encryption(values = [10, 'fam', 'hello world']):
    cls = c.module('key')
    for value in values:
        value = str(value)
        key = cls.new_key(crypto_type=crypto_type)
        enc = key.encrypt(value)
        dec = key.decrypt(enc)
        assert dec == value, f'encryption failed, {dec} != {value}'
    return {'encrypted':enc, 'decrypted': dec}

def test_encryption_with_password(value = 10, password = 'fam'):
    cls = c.module('key')
    value = str(value)
    key = cls.new_key(crypto_type=crypto_type)
    enc = key.encrypt(value, password=password)
    dec = key.decrypt(enc, password=password)
    assert dec == value, f'encryption failed, {dec} != {value}'
    return {'encrypted':enc, 'decrypted': dec}

def test_key_encryption(test_key='test.key'):
    self = c.module('key')
    key = self.add_key(test_key, refresh=True, crypto_type=crypto_type)
    og_key = self.get_key(test_key, crypto_type=crypto_type)
    r = self.encrypt_key(test_key)
    self.decrypt_key(test_key, password=r['password'])
    key = self.get_key(test_key, crypto_type=crypto_type)
    assert key.ss58_address == og_key.ss58_address, f'key encryption failed, {key.ss58_address} != {self.ss58_address}'
    return {'success': True, 'msg': 'test_key_encryption passed'}

def test_key_management(key1='test.key' , key2='test2.key'):
    self = c.module('key')
    if self.key_exists(key1):
        self.rm_key(key1)
    if self.key_exists(key2):
        self.rm_key(key2)
    self.add_key(key1, crypto_type=crypto_type)
    k1 = self.get_key(key1, crypto_type=crypto_type)
    assert self.key_exists(key1), f'Key management failed, key still exists'
    self.mv_key(key1, key2, crypto_type=crypto_type)
    k2 = self.get_key(key2, crypto_type=crypto_type)
    assert k1.ss58_address == k2.ss58_address, f'Key management failed, {k1.ss58_address} != {k2.ss58_address}'
    assert self.key_exists(key2), f'Key management failed, key does not exist'
    assert not self.key_exists(key1), f'Key management failed, key still exists'
    self.mv_key(key2, key1, crypto_type=crypto_type)
    assert self.key_exists(key1), f'Key management failed, key does not exist'
    assert not self.key_exists(key2), f'Key management failed, key still exists'
    self.rm_key(key1)
    # self.rm_key(key2)
    assert not self.key_exists(key1), f'Key management failed, key still exists'
    assert not self.key_exists(key2), f'Key management failed, key still exists'
    return {'success': True, 'msg': 'test_key_management passed'}


def test_signing():
    self = c.module('key')(crypto_type=crypto_type)
    sig = self.sign('test')
    assert self.verify('test',sig, self.public_key)
    return {'success':True}

def test_key_encryption(password='1234'):
    cls = c.module('key')
    path = 'test.enc'
    cls.add_key('test.enc', refresh=True, crypto_type=crypto_type)
    assert cls.is_key_encrypted(path) == False, f'file {path} is encrypted'
    cls.encrypt_key(path, password=password)
    assert cls.is_key_encrypted(path) == True, f'file {path} is not encrypted'
    cls.decrypt_key(path, password=password)
    assert cls.is_key_encrypted(path) == False, f'file {path} is encrypted'
    cls.rm(path)
    print('file deleted', path, c.exists, 'fam')
    assert not c.exists(path), f'file {path} not deleted'
    return {'success': True, 'msg': 'test_key_encryption passed'}

def test_move_key():
    self = c.module('key')()
    self.add_key('testfrom', crypto_type=crypto_type)
    assert self.key_exists('testfrom')
    og_key = self.get_key('testfrom', crypto_type=crypto_type)
    self.mv_key('testfrom', 'testto', crypto_type=crypto_type)
    assert self.key_exists('testto', crypto_type=crypto_type)
    assert not self.key_exists('testfrom')
    new_key = self.get_key('testto', crypto_type=crypto_type)
    assert og_key.ss58_address == new_key.ss58_address
    self.rm_key('testto')
    assert not self.key_exists('testto')
    return {'success':True, 'msg':'test_move_key passed', 'key':new_key.ss58_address}


def test_ss58_encoding():
    self = c.module('key')
    keypair = self.create_from_uri('//Alice')
    ss58_address = keypair.ss58_address
    public_key = keypair.public_key
    assert keypair.ss58_address == self.ss58_encode(public_key, ss58_format=42)
    assert keypair.ss58_address == self.ss58_encode(public_key, ss58_format=42)
    assert keypair.public_key.hex() == self.ss58_decode(ss58_address)
    assert keypair.public_key.hex() == self.ss58_decode(ss58_address)
    return {'success':True}


