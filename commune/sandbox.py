from substrateinterface import SubstrateInterface, Keypair
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

Key1 = Keypair.create_from_uri('//Alice')
Key2 = Keypair.create_from_uri('//Bob')
# Generate a private key for the server
server_private_key = Key1.private_key.hex()
server_public_key = Key1.public_key.hex()
# Generate a private key for the peer
peer_private_key = Key2.private_key.hex()
peer_public_key = Key2.public_key.hex()    

# Perform key exchange
shared_key = server_private_key.key_exchange(peer_public_key)

# Perform key derivation
derived_key = HKDF(
    algorithm=hashes.SHA256(),
    length=32,
    salt=None,
    info=b'handshake data',
).derive(shared_key)

# Demonstrate that the handshake performed in the opposite direction gives the same final value
same_shared_key = peer_private_key.key_exchange(server_public_key)
same_derived_key = HKDF(
    algorithm=hashes.SHA256(),
    length=32,
    salt=None,
    info=b'handshake data',
).derive(same_shared_key)

assert derived_key == same_derived_key
print("Key exchange successful!")