import base64
import hashlib
from Crypto import Random
from Crypto.Cipher import AES
from copy import deepcopy
import json
import sys
import inspect
import time
import commune as c


class AESKey(c.Module):

    def __init__(self, key:str = 'dummy' ): 
        self.set_password(key)

    def set_password(self, key:str):
        if isinstance(key, str):
            key = key.encode()
        self.bs = AES.block_size
        self.key_phrase = hashlib.sha256(key).digest()
        return {'msg': 'set the password'}

    def encrypt(self, data, return_string = True):
        data = c.python2str(data)
        data = self._pad(data)
        iv = Random.new().read(AES.block_size)
        cipher = AES.new(self.key_phrase, AES.MODE_CBC, iv)
        encrypted_bytes = base64.b64encode(iv + cipher.encrypt(data.encode()))
        encrypted_data =  encrypted_bytes.decode() if return_string else encrypted_bytes
        return encrypted_data

    def decrypt(self, enc):
        enc = base64.b64decode(enc)
        iv = enc[:AES.block_size]
        cipher = AES.new(self.key_phrase, AES.MODE_CBC, iv)
        decrypted_data =  self._unpad(cipher.decrypt(enc[AES.block_size:])).decode('utf-8')
        return self.str2python(decrypted_data)

    def _pad(self, s):
        return s + (self.bs - len(s) % self.bs) * chr(self.bs - len(s) % self.bs)

    def _unpad(self, s):
        return s[:-ord(s[len(s)-1:])]
    
   