import random
import math
import base64
from typing import Tuple, Union, Optional

class RSA:
    """
    A comprehensive RSA implementation class that provides key generation,
    encryption, decryption, signing, and verification functionality.
    """
    
    def __init__(self, path='key',  key_size: int = 2048):
        """
        Initialize the RSA object with optional key size.
        
        Args:
            key_size: Bit length of the RSA key (default: 2048)
        """
        self.key_size = key_size
        self.public_key = None
        self.private_key = None


    def get_path(self, path):
        return c.storage_path + '/rsa/' + path

    
    def generate_keys(self) -> Tuple[Tuple, Tuple[int, int]]:
        """
        Generate a new RSA key pair.
        
        Returns:
            Tuple containing public key (e, n) and private key (d, n)
        """
        # Generate two large prime numbers
        p = self._generate_large_prime(self.key_size // 2)
        q = self._generate_large_prime(self.key_size // 2)
        
        # Calculate n = p * q
        n = p * q
        
        # Calculate Euler's totient function: φ(n) = (p-1) * (q-1)
        phi = (p - 1) * (q - 1)
        
        # Choose public exponent e (commonly 65537)
        e = 65537
        
        # Ensure e is coprime to phi
        while math.gcd(e, phi) != 1:
            e += 2
        
        # Calculate private exponent d (modular multiplicative inverse of e mod phi)
        d = self._mod_inverse(e, phi)
        
        self.public_key = (e, n)
        self.private_key = (d, n)
        
        return self.public_key, self.private_key
    
    def encrypt(self, message: Union[str, int, bytes], public_key: Optional[Tuple[int, int]] = None) -> int:
        """
        Encrypt a message using RSA.
        
        Args:
            message: The message to encrypt (string, integer, or bytes)
            public_key: Optional public key tuple (e, n). Uses stored key if None.
            
        Returns:
            Encrypted message as an integer
        """
        if public_key is None:
            if self.public_key is None:
                raise ValueError("No public key available. Generate keys first or provide a public key.")
            public_key = self.public_key
        
        e, n = public_key
        
        # Convert message to integer if it's a string or bytes
        if isinstance(message, str):
            message = int.from_bytes(message.encode(), byteorder='big')
        elif isinstance(message, bytes):
            message = int.from_bytes(message, byteorder='big')
        
        # Check if message is too large
        if message >= n:
            raise ValueError("Message is too large for the key size")
        
        # Encrypt: c = m^e mod n
        ciphertext = pow(message, e, n)
        return ciphertext
    
    def decrypt(self, ciphertext: int, private_key: Optional[Tuple[int, int]] = None) -> int:
        """
        Decrypt a ciphertext using RSA.
        
        Args:
            ciphertext: The encrypted message (integer)
            private_key: Optional private key tuple (d, n). Uses stored key if None.
            
        Returns:
            Decrypted message as an integer
        """
        if private_key is None:
            if self.private_key is None:
                raise ValueError("No private key available. Generate keys first or provide a private key.")
            private_key = self.private_key
        
        d, n = private_key
        
        # Decrypt: m = c^d mod n
        message = pow(ciphertext, d, n)
        return message
    
    def decrypt_to_string(self, ciphertext: int, private_key: Optional[Tuple] = None) -> str:
        """
        Decrypt a ciphertext and convert the result to a string.
        
        Args:
            ciphertext: The encrypted message (integer)
            private_key: Optional private key tuple (d, n). Uses stored key if None.
            
        Returns:
            Decrypted message as a string
        """
        decrypted = self.decrypt(ciphertext, private_key)
        
        # Convert integer to bytes, then to string
        byte_length = (decrypted.bit_length() + 7) // 8
        decrypted_bytes = decrypted.to_bytes(byte_length, byteorder='big')
        
        # Remove any null bytes
        decrypted_bytes = decrypted_bytes.lstrip(b'\x00')
        
        return decrypted_bytes.decode('utf-8', errors='ignore')
    
    def sign(self, message: Union[str, int, bytes], private_key: Optional[Tuple] = None) -> int:
        """
        Sign a message using RSA.
        
        Args:
            message: The message to sign (string, integer, or bytes)
            private_key: Optional private key tuple (d, n). Uses stored key if None.
            
        Returns:
            Digital signature as an integer
        """
        if private_key is None:
            if self.private_key is None:
                raise ValueError("No private key available. Generate keys first or provide a private key.")
            private_key = self.private_key
        
        d, n = private_key
        
        # Convert message to integer if it's a string or bytes
        if isinstance(message, str):
            message = int.from_bytes(message.encode(), byteorder='big')
        elif isinstance(message, bytes):
            message = int.from_bytes(message, byteorder='big')
        
        # Hash the message (simple implementation - in practice, use a proper hash function)
        hashed = message % n
        
        # Sign: s = hash(m)^d mod n
        signature = pow(hashed, d, n)
        return signature
    
    def verify(self, message: Union, signature: int, 
               public_key: Optional[Tuple] = None) -> bool:
        """
        Verify a digital signature using RSA.
        
        Args:
            message: The original message (string, integer, or bytes)
            signature: The digital signature to verify
            public_key: Optional public key tuple (e, n). Uses stored key if None.
            
        Returns:
            True if signature is valid, False otherwise
        """
        if public_key is None:
            if self.public_key is None:
                raise ValueError("No public key available. Generate keys first or provide a public key.")
            public_key = self.public_key
        
        e, n = public_key
        
        # Convert message to integer if it's a string or bytes
        if isinstance(message, str):
            message = int.from_bytes(message.encode(), byteorder='big')
        elif isinstance(message, bytes):
            message = int.from_bytes(message, byteorder='big')
        
        # Hash the message (simple implementation - in practice, use a proper hash function)
        hashed = message % n
        
        # Verify: hash(m) == s^e mod n
        decrypted_signature = pow(signature, e, n)
        return decrypted_signature == hashed
    
    def export_public_key(self) -> str:
        """
        Export the public key in a base64-encoded format.
        
        Returns:
            Base64-encoded public key
        """
        if self.public_key is None:
            raise ValueError("No public key available. Generate keys first.")
        
        e, n = self.public_key
        key_data = f"{e}:{n}".encode()
        return base64.b64encode(key_data).decode()
    
    def export_private_key(self) -> str:
        """
        Export the private key in a base64-encoded format.
        
        Returns:
            Base64-encoded private key
        """
        if self.private_key is None:
            raise ValueError("No private key available. Generate keys first.")
        
        d, n = self.private_key
        key_data = f"{d}:{n}".encode()
        return base64.b64encode(key_data).decode()
    
    def import_public_key(self, key_data: str) -> None:
        """
        Import a public key from a base64-encoded string.
        
        Args:
            key_data: Base64-encoded public key
        """
        decoded = base64.b64decode(key_data).decode()
        e, n = map(int, decoded.split(':'))
        self.public_key = (e, n)
    
    def import_private_key(self, key_data: str) -> None:
        """
        Import a private key from a base64-encoded string.
        
        Args:
            key_data: Base64-encoded private key
        """
        decoded = base64.b64decode(key_data).decode()
        d, n = map(int, decoded.split(':'))
        self.private_key = (d, n)
    
    def _is_prime(self, n: int, k: int = 5) -> bool:
        """
        Miller-Rabin primality test.
        
        Args:
            n: Number to test for primality
            k: Number of test rounds
            
        Returns:
            True if n is probably prime, False otherwise
        """
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0:
            return False
        
        # Write n as 2^r * d + 1
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        
        # Witness loop
        for _ in range(k):
            a = random.randint(2, n - 2)
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True
    
    def _generate_large_prime(self, bits: int) -> int:
        """
        Generate a large prime number with the specified bit length.
        
        Args:
            bits: Bit length of the prime number
            
        Returns:
            A prime number
        """
        while True:
            # Generate a random odd number with the specified bit length
            p = random.getrandbits(bits) | (1 << bits - 1) | 1
            if self._is_prime(p):
                return p
    
    def _extended_gcd(self, a: int, b: int) -> Tuple[int, int, int]:
        """
        Extended Euclidean Algorithm to find gcd and coefficients.
        
        Args:
            a, b: Integers
            
        Returns:
            Tuple (gcd, x, y) such that ax + by = gcd
        """
        if a == 0:
            return b, 0, 1
        else:
            gcd, x, y = self._extended_gcd(b % a, a)
            return gcd, y - (b // a) * x, x
    
    def _mod_inverse(self, a: int, m: int) -> int:
        """
        Calculate the modular multiplicative inverse of a modulo m.
        
        Args:
            a: Integer
            m: Modulus
            
        Returns:
            Integer b such that (a * b) % m == 1
        """
        gcd, x, y = self._extended_gcd(a, m)
        if gcd != 1:
            raise ValueError("Modular inverse does not exist")
        else:
            return x % m