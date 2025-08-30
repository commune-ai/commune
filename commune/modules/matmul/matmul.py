import numpy as np
import hashlib
import time
import base64
from PIL import Image
import io
import time
import commune as c

class Matmul:
    """
    A Proof of Work system based on matrix multiplication.
    This can be used for computational tasks like image processing.
    """
    
    def __init__(self):
        """
        Initialize the PoW system with a specified difficulty.
        
        Args:
            difficulty (int): The number of leading zeros required in the hash.
        """
        self.serializer = c.module('serializer')()
    def generate_params(self, size=64):
        """
        Generate two random matrices for the challenge.
        
        Args:
            size (int): The size of the matrices (size x size).
            
        Returns:
            tuple: Two random matrices A and B.
        """
        a = np.random.rand(size, size)
        b = np.random.rand(size, size)
        return a, b
    
    def compute_proof(self, a, b, key=None):
        """
        Perform the matrix multiplication with a nonce and compute the hash.
        
        Args:
            a (ndarray): First matrix.
            b (ndarray): Second matrix.
            nonce (int): A value to modify the result.
            
        Returns:
            tuple: (result_matrix, hash_result, nonce)
        """
        # Perform matrix multiplication
        print(a, b, type(b))
        result = np.matmul(a, b)

        key = c.get_key(key)
        
        data =  {
            'params': {'a': a, 'b': b},
            'result': result,
            'time': time.time(),
            'key': key.key_address
        }
        data = self.serializer.serialize(data)
        data['signature'] = c.sign(data, key=key)

        return data

    def verify_proof(self, proof, key=None):
        """
        Verify that a given nonce produces the expected hash.
        
        Args:
            a (ndarray): First matrix.
            b (ndarray): Second matrix.
            nonce (int): The nonce to verify.
            hash_result (str): The expected hash result.
            
        Returns:
            bool: True if the proof is valid, False otherwise.
        """
        signature = proof.pop('signature')        
        assert c.verify(proof, signature, proof['key'])
        proof = self.serializer.deserialize(proof)
        new_proof = self.compute_proof(**proof['params'], key=key)
        new_proof = self.serializer.deserialize(proof)
        old_result_hash = self.hash_matrix(proof['result'])
        new_result_hash = self.hash_matrix(new_proof['result'])
        return bool(old_result_hash == new_result_hash)

    def hash_matrix(self, data:'np.ndarray')-> bytes:
        import msgpack_numpy
        import msgpack
        output = msgpack.packb(data, default=msgpack_numpy.encode)
        return c.hash(output)
    
    def test(self):
        """
        Run a simple test of the PoW system.
        """
        a, b = self.generate_params(size=4)
        proof= self.compute_proof(a, b)        
        verified =  self.verify_proof(proof)
        assert verified
        return {'verified': verified, 'proof': proof}