import base64
import hashlib
from Crypto import Random
from Crypto.Cipher import AES
from copy import deepcopy
import json
import sys
import inspect
import time
class AESKey:

    def __init__(self, key): 
        self.bs = AES.block_size
        self.key = hashlib.sha256(key.encode()).digest()

    @staticmethod
    def python2str(input):
        input = deepcopy(input)
        input_type = type(input)
        if input_type in [dict]:
            input = json.dumps(input)
        elif input_type in [list, tuple, set]:
            input = json.dumps(list(input))
        elif input_type in [int, float, bool]:
            input = str(input)
        return input

    @staticmethod
    def str2python(input)-> dict:
        assert isinstance(input, str)
        try:
            output_dict = json.loads(input)
        except json.JSONDecodeError as e:
            return input

        return output_dict


    def encrypt(self, raw, return_string = True):
        raw = self.python2str(raw)
        raw = self._pad(raw)
        iv = Random.new().read(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        
        encrypted_bytes = base64.b64encode(iv + cipher.encrypt(raw.encode()))
        encrypted_data =  encrypted_bytes.decode() if return_string else encrypted_bytes

        return encrypted_data

    def decrypt(self, enc):
        enc = base64.b64decode(enc)
        iv = enc[:AES.block_size]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        decrypted_data =  self._unpad(cipher.decrypt(enc[AES.block_size:])).decode('utf-8')
        return self.str2python(decrypted_data)

    def _pad(self, s):
        return s + (self.bs - len(s) % self.bs) * chr(self.bs - len(s) % self.bs)

    @staticmethod
    def _unpad(s):
        return s[:-ord(s[len(s)-1:])]


    @classmethod
    def test_encrypt_decrypt(cls, key='dummy'):
        import streamlit as st
        print(inspect.stack()[0][3])
        self = cls(key=key)
        test_objects = [
            [1,2,3,5],
            {'fam': 1, 'bro': 'fam', 'chris': {'sup': [1,'dawg']}},
            1,
            'fam', 
        ]
        import time
        for test_object in test_objects:
            start_time = time.clock()
            encrypted = self.encrypt(test_object)
            decrypted = self.decrypt(encrypted)
            assert decrypted == test_object, f'FAILED: {test_encrypt_decrypt} {test_object} FAILED'
            
            size_bytes = sys.getsizeof(test_object)
            seconds =  time.clock() - start_time
            rate = size_bytes / seconds

        print('PASSED test_encrypt_decrypt')

        return True
    
    



    @classmethod
    def test_encrypt_decrypt_throughput(cls, key='dummy'):
        import streamlit as st
        print(inspect.stack()[0][3])
        self = cls(key=key)
        test_object = [1,2,3,5]*1000000
        start_time = time.clock()
        encrypted = self.encrypt(test_object)
        seconds =  time.clock() - start_time        
        size_bytes = sys.getsizeof(test_object)
        encrypt_rate = size_bytes / seconds

        start_time = time.clock()
        decrypted = self.decrypt(encrypted)
        seconds =  time.clock() - start_time        
        size_bytes = sys.getsizeof(test_object)
        decrypt_rate = size_bytes / seconds


        st.write(f'ENCRYPT SPEED (MB per Second): {encrypt_rate//1000}')
        st.write(f'DECRYPT SPEED (MB per Second): {decrypt_rate//1000}')

        print('PASSED test_encrypt_decrypt')

        return True
    




    @classmethod
    def test(cls):
        import streamlit as st
        for attr in dir(cls):
            if attr[:len('test_')] == 'test_':
                getattr(cls, attr)()
                st.write('PASSED',attr)


    @classmethod
    def streamlit(cls):
        import streamlit as st
        with st.expander('Tests'):
            cls.test()
        

if __name__ =='__main__':
    AESKey.streamlit()
