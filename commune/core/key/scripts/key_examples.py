#!/usr/bin/env python3
# Example script demonstrating key module functionality

import commune as c

def basic_key_operations():
    print("\n=== Basic Key Operations ===")
    # Create a new key
    key = c.key()
    print(f"Generated new key: {key.key_address}")
    
    # Create keys with different crypto types
    sr25519_key = c.key(crypto_type='sr25519')
    ed25519_key = c.key(crypto_type='ed25519')
    ecdsa_key = c.key(crypto_type='ecdsa')
    
    print(f"SR25519 key: {sr25519_key.key_address}")
    print(f"ED25519 key: {ed25519_key.key_address}")
    print(f"ECDSA key: {ecdsa_key.key_address}")

def signing_and_verification():
    print("\n=== Signing and Verification ===")
    key = c.key()
    
    # Data to sign
    data = "Hello, Commune!"
    print(f"Data to sign: {data}")
    
    # Sign the data
    signature = key.sign(data)
    print(f"Signature: {signature.hex() if isinstance(signature, bytes) else signature}")
    
    # Verify the signature
    is_valid = key.verify(data, signature)
    print(f"Signature valid: {is_valid}")
    
    # Try with invalid data
    invalid_data = "Modified data"
    is_valid = key.verify(invalid_data, signature)
    print(f"Invalid data verification: {is_valid} (should be False)")

def encryption_and_decryption():
    print("\n=== Encryption and Decryption ===")
    key = c.key()
    
    # Data to encrypt
    data = "Secret message for Commune"
    print(f"Original data: {data}")
    
    # Encrypt the data
    encrypted = key.encrypt(data)
    print(f"Encrypted: {encrypted}")
    
    # Decrypt the data
    decrypted = key.decrypt(encrypted)
    print(f"Decrypted: {decrypted}")
    assert data == decrypted, "Decryption failed!"

def mnemonic_operations():
    print("\n=== Mnemonic Operations ===")
    key = c.key()
    
    # Generate a new mnemonic
    mnemonic = key.generate_mnemonic()
    print(f"Generated mnemonic: {mnemonic}")
    
    # Create a key from the mnemonic
    mnemonic_key = c.key(mnemonic=mnemonic)
    print(f"Key from mnemonic: {mnemonic_key.key_address}")

def key_storage():
    print("\n=== Key Storage ===")
    
    # Add a key to storage
    key = c.key()
    key_name = "example_key"
    key.add_key(key_name)
    print(f"Added key '{key_name}' to storage")
    
    # List available keys
    keys = key.keys()
    print(f"Available keys: {keys}")
    
    # Get key from storage
    retrieved_key = key.get_key(key_name)
    print(f"Retrieved key: {retrieved_key.key_address}")
    
    # Clean up - remove the key
    key.rm_key(key_name)
    print(f"Removed key '{key_name}' from storage")

def main():
    print("COMMUNE KEY MODULE EXAMPLES")
    print("==========================")
    
    basic_key_operations()
    signing_and_verification()
    encryption_and_decryption()
    mnemonic_operations()
    key_storage()
    
    print("\nAll examples completed successfully!")

if __name__ == "__main__":
    main()
