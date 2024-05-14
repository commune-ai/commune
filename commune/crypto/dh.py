import commune as c


class DH(c.Module):
    description = 'Diffie-Hellman key exchange'
    
    def __init__(self, public_key1 = 'hey', public_key2='bro', private_key = 'test'):
        self.public_key1 = self.str2int(public_key1)
        self.public_key2 = self.str2int(public_key2)
        self.private_key =  self.str2int(c.hash(c.get_key(private_key).mnemonic) if c.key_exists(private_key) else private_key)
        # convert string to int via binary

        c.print(self.__dict__)

    
        self.full_key = None

    def str2int(self, x):
        nchars = len(x)
        x = sum(ord(x[byte])<<8*(nchars-byte-1) for byte in range(nchars))
        return c.print(x)

        
    def generate_partial_key(self):
        partial_key = self.public_key1**self.private_key
        partial_key = partial_key%self.public_key2
        return partial_key
    
    def generate_full_key(self, partial_key_r):
        full_key = partial_key_r**self.private_key
        full_key = full_key%self.public_key2
        self.full_key = full_key
        return full_key
    
    def encrypt_message(self, message):
        encrypted_message = ""
        key = self.full_key
        for c in message:
            encrypted_message += chr(ord(c)+key)
        return encrypted_message
    
    def decrypt_message(self, encrypted_message):
        decrypted_message = ""
        key = self.full_key
        for c in encrypted_message:
            decrypted_message += chr(ord(c)-key)
        return decrypted_message
    

    @classmethod
    def test(cls, public_key1='hey', public_key2='bro'):
        dh1 = cls(public_key1=public_key1, public_key2=public_key1, private_key='test1')
        dh2 = cls(public_key1=public_key1, public_key2=public_key1, private_key='test2')
        partial_key1 = dh1.generate_partial_key()
        partial_key2 = dh2.generate_partial_key()
        full_key1 = dh1.generate_full_key(partial_key2)
        full_key2 = dh2.generate_full_key(partial_key1)
        message = 'hello'
        encrypted_message = dh1.encrypt_message(message)
        decrypted_message = dh2.decrypt_message(encrypted_message)
        c.print(f'message: {message}')
        c.print(f'encrypted_message: {encrypted_message}')
        c.print(f'decrypted_message: {decrypted_message}')
        c.print(f'full_key1: {full_key1}')
        c.print(f'full_key2: {full_key2}')
        c.print(f'partial_key1: {partial_key1}')
        c.print(f'partial_key2: {partial_key2}')
        assert message == decrypted_message
        assert full_key1 == full_key2
        assert partial_key1 == partial_key2
        c.print('SUCCESS', color='green')
