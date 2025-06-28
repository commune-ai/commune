from Crypto import Random
import hashlib
from Crypto.Cipher import AES
import copy
import base64
import json
from typing import *

class AesKey:
    """
    AES encryption and decryption class.
    """

    def __init__(self, password):
        self.set_password(password)

    def encrypt(self, data, password=None):
        password = self.get_password(password)  
        data = copy.deepcopy(data)
        if not isinstance(data, str):
            data = str(data)
        data = data + (AES.block_size - len(data) % AES.block_size) * chr(AES.block_size - len(data) % AES.block_size)
        iv = Random.new().read(AES.block_size)
        cipher = AES.new(password, AES.MODE_CBC, iv)
        encrypted_bytes = base64.b64encode(iv + cipher.encrypt(data.encode()))
        return encrypted_bytes.decode() 


    def data2str(self, data: Union['ScaleBytes', bytes, str]) -> str:
        if not isinstance(data, str):
            data = json.dumps(data)
        return data

    def decrypt(self, data, password:str=None):  
        password = self.get_password(password)  
        data = base64.b64decode(data)
        iv = data[:AES.block_size]
        cipher = AES.new(password, AES.MODE_CBC, iv)
        data =  cipher.decrypt(data[AES.block_size:])
        data = data[:-ord(data[len(data)-1:])].decode('utf-8')
        data = self.str2data(data)
        return data

    def test(self,  values = [10, 'fam', 'hello world'], password='1234'):
        if isinstance(crypto_type, list):
            return [self.test_encryption(values=values, crypto_type=k) for k in crypto_type]
        for value in values:
            value = str(value)
            key = Key(crypto_type=crypto_type)
            enc = key.encrypt(value, password)
            dec = key.decrypt(enc, password)
            assert dec == value, f'encryption failed, {dec} != {value}'
        return {'encrypted':enc, 'decrypted': dec, 'crypto_type':key.crypto_type}


    def str2data(self, data: Union['ScaleBytes', bytes, str]) -> Union['ScaleBytes', bytes, str]:
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                pass
        return data

    def set_password(self, password):
        self.password=self.get_password(password)
        return {'msg': 'password set'}

    def get_password(self, password:str=None):
        if password == None: 
            password =  self.password
        if isinstance(password, str):
            password = password.encode()
        # if password is a key, use the key's private key as password
        return hashlib.sha256(password).digest()
