import commune as c
import os
import base64
import hmac
import hashlib
import secrets
from typing import Optional
from Crypto.Cipher import AES

class DiffieHellman:
    """
    Safe Diffie–Hellman over RFC 3526 MODP groups with HKDF + AES-GCM.
    """

    def __init__(self, key: 'Key' = None, group_size: int = 2048, crypto_type: str = 'sr25519'):
        if group_size == 2048:
            self.p = int(
                'FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1'
                '29024E088A67CC74020BBEA63B139B22514A08798E3404DD'
                'EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245'
                'E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED'
                'EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3D'
                'C2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F'
                '83655D23DCA3AD961C62F356208552BB9ED529077096966D'
                '670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B'
                'E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9'
                'DE2BCBF6955817183995497CEA956AE515D2261898FA0510'
                '15728E5A8AACAA68FFFFFFFFFFFFFFFF', 16
            )
        elif group_size == 4096:
            self.p = int(
                'FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1'
                '29024E088A67CC74020BBEA63B139B22514A08798E3404DD'
                'EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245'
                'E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED'
                'EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3D'
                'C2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F'
                '83655D23DCA3AD961C62F356208552BB9ED529077096966D'
                '670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B'
                'E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9'
                'DE2BCBF6955817183995497CEA956AE515D2261898FA0510'
                '15728E5A8AAAC42DAD33170D04507A33A85521ABDF1CBA64'
                'ECFB850458DBEF0A8AEA71575D060C7DB3970F85A6E1E4C7'
                'ABF5AE8CDB0933D71E8C94E04A25619DCEE3D2261AD2EE6B'
                'F12FFA06D98A0864D87602733EC86A64521F2B18177B200C'
                'BBE117577A615D6C770988C0BAD946E208E24FA074E5AB31'
                '43DB5BFCE0FD108E4B82D120A93AD2CAFFFFFFFFFFFFFFFF', 16
            )
        else:
            raise ValueError("Unsupported group size. Use 2048 or 4096.")
        self.g = 2
        self.group_size = group_size
        self.q = (self.p - 1) // 2  # safe prime subgroup order
        self.key = self._get_key(key, crypto_type=crypto_type)

        # Private exponent in [2, q-2]
        self._x = self._generate_private_exponent()
        self._Y = pow(self.g, self._x, self.p)

    # --- Key / RNG helpers ---

    def _get_key(self, key, crypto_type='sr25519'):
        # Kept for compatibility with your codebase; not used for RNG by default.
        if key is None and hasattr(self, 'key'):
            return self.key
        return c.get_key(key, crypto_type=crypto_type)

    def _generate_private_exponent(self) -> int:
        # Use CSPRNG. If you *must* tie to identity, derive with HKDF(PRK=key_bytes).
        return 2 + secrets.randbelow(self.q - 3)

    # --- Public API ---

    def get_public_value(self) -> int:
        return self._Y

    # --- Core math ---

    def _validate_peer_public(self, Y: int) -> None:
        if not (2 <= Y <= self.p - 2):
            raise ValueError("Peer public value out of range")
        # Subgroup check: Y^q ≡ 1 mod p for safe primes
        if pow(Y, self.q, self.p) != 1:
            raise ValueError("Peer public value not in prime-order subgroup")

    def compute_shared_secret(self, other_public_value: int) -> bytes:
        self._validate_peer_public(other_public_value)
        Z = pow(other_public_value, self._x, self.p)  # raw DH shared secret (int)
        # Serialize with fixed length to avoid length leaks
        zs = Z.to_bytes((self.p.bit_length() + 7) // 8, 'big')
        return zs

    # --- HKDF (RFC 5869) ---

    @staticmethod
    def _hkdf_extract(salt: Optional[bytes], ikm: bytes, hashmod=hashlib.sha256) -> bytes:
        if salt is None:
            salt = b'\x00' * hashmod().digest_size
        return hmac.new(salt, ikm, hashmod).digest()

    @staticmethod
    def _hkdf_expand(prk: bytes, info: bytes, L: int, hashmod=hashlib.sha256) -> bytes:
        n = (L + hashmod().digest_size - 1) // hashmod().digest_size
        if n > 255:
            raise ValueError("Cannot expand to more than 255 blocks")
        okm = b""
        T = b""
        for i in range(1, n + 1):
            T = hmac.new(prk, T + info + bytes([i]), hashmod).digest()
            okm += T
        return okm[:L]

    def derive_key(self, other_public_value: int, *, salt: Optional[bytes] = None,
                   info: Optional[bytes] = None, key_length: int = 32) -> bytes:
        if info is None:
            info = b"DH-MODP-HKDF-AESGCM"
        shared = self.compute_shared_secret(other_public_value)
        prk = self._hkdf_extract(salt, shared)
        return self._hkdf_expand(prk, info, key_length)

    # --- Encrypt / Decrypt (AES-GCM, 128-bit tag) ---

    def encrypt(self, data, other_public_value: int, *, aad: Optional[bytes] = None) -> str:
        if not isinstance(data, (bytes, bytearray)):
            data = str(data).encode('utf-8')
        # Fresh salt per session (helps with key separation if reuse occurs)
        salt = os.urandom(32)
        key = self.derive_key(other_public_value, salt=salt, key_length=32)
        nonce = os.urandom(12)  # 96-bit nonce recommended for GCM
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce, mac_len=16)
        if aad:
            cipher.update(aad)
        ct, tag = cipher.encrypt_and_digest(data)
        # Pack: salt || nonce || tag || ciphertext
        blob = salt + nonce + tag + ct
        return base64.b64encode(blob).decode('ascii')

    def decrypt(self, token_b64: str, other_public_value: int, *, aad: Optional[bytes] = None) -> str:
        blob = base64.b64decode(token_b64)
        if len(blob) < 32 + 12 + 16:
            raise ValueError("Ciphertext too short")
        salt, rest = blob[:32], blob[32:]
        nonce, rest = rest[:12], rest[12:]
        tag, ct = rest[:16], rest[16:]
        key = self.derive_key(other_public_value, salt=salt, key_length=32)
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce, mac_len=16)
        if aad:
            cipher.update(aad)
        pt = cipher.decrypt_and_verify(ct, tag)
        return pt.decode('utf-8')

    # --- Self-test ---



    def hand_shake(self,public_value:str) -> bool:
        A = self.get_public_value()
        B = int(public_value)

        a_secret = self.compute_shared_secret(B)
        b_secret = public_key.compute_shared_secret(A)
        return a_secret == b_secret
    @staticmethod
    def test():
        alice = DiffieHellman('alice')
        bob = DiffieHellman('bob')
        A = alice.get_public_value()
        B = bob.get_public_value()

        a_secret = alice.compute_shared_secret(B)
        b_secret = bob.compute_shared_secret(A)
        secrets_match = (a_secret == b_secret)

        msg = "This is a secret message for testing"
    
        t0 = c.time()
        enc = alice.encrypt(msg, B, aad=b"demo")
        dec = bob.decrypt(enc, A, aad=b"demo")
        t2 = c.time()

        return {
            "success": secrets_match and (msg == dec),
            "secrets_match": secrets_match,
            "encryption_works": (msg == dec),
            "alice_public": A,
            "bob_public": B,
            "test_message": msg,
            "encrypted": enc,
            "decrypted": dec,
            "alice_address": alice.key.address,
            "bob_address": bob.key.address,
            "enc_dec_loop": t2 - t0
        }
