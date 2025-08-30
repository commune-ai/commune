import commune as c

from .key import Key

class TestKey:
    def test_key_storage(self):
        key = Key()
        key.add_key('test', refresh=True)
        assert key.key_exists('test'), 'Key not found'
        key2 = key.get_key('test')
        assert key.address == key.address, f'{key.address} != {key2.address}'
        key.remove_key('test')
        assert not key.key_exists('test'), 'Key not removed'
        return {'status': 'success', 'message': 'Key storage test passed', 'keys': [key, key2]}

    def test_verify(self):
        key = Key()
        key2 = Key()
        data = key.resolve_message('Hello, World!')
        signature = key.sign(data)
        assert key.verify(data, signature=signature['signature'], vrs=signature['vrs'], address=key.address)
        assert key.verify(data, signature=signature['signature'], vrs=signature['vrs'], address=signature['address'])
        assert not key.verify(data, 
                                signature=signature['signature'], 
                                vrs=signature['vrs'], 
                                address=key2.address)
        
    def test_encrypt(self):
        # test encryption
        password = 'password'
        key = Key()
        key2 = Key()
        data = 'Hello, World!'
        encrypted_data = key.encrypt(data, password)
        decrypted_data = key.decrypt(encrypted_data, password)
        assert data == decrypted_data, f'{data} != {decrypted_data}'
        encrypted_data = key.encrypt(data)
        decrypted_data = key.decrypt(encrypted_data)
        assert data == decrypted_data, f'{data} != {decrypted_data}'
        return {'status': 'success', 'message': 'Key test passed', 'keys': [key, key2]}

    def test_key_loading(self):
        key = Key()
        key_dict = key.to_dict()
        key2 = key.from_dict(key_dict)
        assert key.address == key2.address, f'{key.address} != {key2.address}'
        return {'status': 'success', 'message': 'Key test passed', 'keys': [key, key2]}

    # Return True if all tests passed, False otherwise
