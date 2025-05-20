#!/usr/bin/env python3
# Simple test script for key module

import commune as c
import unittest

class TestKeyModule(unittest.TestCase):
    
    def test_key_creation(self):
        # Test default key creation
        key = c.key()
        self.assertIsNotNone(key.key_address)
        self.assertIsNotNone(key.public_key)
        self.assertIsNotNone(key.private_key)
        
        # Test different crypto types
        sr25519_key = c.key(crypto_type='sr25519')
        ed25519_key = c.key(crypto_type='ed25519')
        ecdsa_key = c.key(crypto_type='ecdsa')
        
        self.assertEqual(sr25519_key.crypto_type_name, 'sr25519')
        self.assertEqual(ed25519_key.crypto_type_name, 'ed25519')
        self.assertEqual(ecdsa_key.crypto_type_name, 'ecdsa')
    
    def test_mnemonic(self):
        # Generate mnemonic
        key = c.key()
        mnemonic = key.generate_mnemonic()
        self.assertTrue(key.is_mnemonic(mnemonic))
        
        # Create key from mnemonic
        mnemonic_key = c.key(mnemonic=mnemonic)
        self.assertIsNotNone(mnemonic_key.key_address)
    
    def test_signing_verification(self):
        key = c.key()
        data = "Test data for signing"
        
        # Test signing
        signature = key.sign(data)
        self.assertIsNotNone(signature)
        
        # Test verification
        is_valid = key.verify(data, signature)
        self.assertTrue(is_valid)
        
        # Test invalid verification
        is_valid = key.verify("Modified data", signature)
        self.assertFalse(is_valid)
    
    def test_encryption_decryption(self):
        key = c.key()
        data = "Secret test data"
        
        # Test encryption
        encrypted = key.encrypt(data)
        self.assertIsNotNone(encrypted)
        
        # Test decryption
        decrypted = key.decrypt(encrypted)
        self.assertEqual(data, decrypted)
    
    def test_key_storage(self):
        key = c.key()
        key_name = "test_key_" + c.random_word()
        
        # Test adding key
        key.add_key(key_name)
        self.assertTrue(key.key_exists(key_name))
        
        # Test retrieving key
        retrieved_key = key.get_key(key_name)
        self.assertEqual(key.key_address, retrieved_key.key_address)
        
        # Test removing key
        key.rm_key(key_name)
        self.assertFalse(key.key_exists(key_name))

if __name__ == "__main__":
    unittest.main()
