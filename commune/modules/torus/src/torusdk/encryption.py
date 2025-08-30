import base64
import hashlib
import json
from typing import Any

from nacl.secret import SecretBox
from nacl.utils import random


class PasswordNotProvidedError(Exception):
    pass


def derive_key(password: str):
    # Derive a 256-bit key from the password using Blake2b
    key = hashlib.blake2b(password.encode(), digest_size=32).digest()
    return key


def encrypt_data(password: str, data: Any) -> tuple[str, int]:
    key = derive_key(password)
    box = SecretBox(key)
    nonce = random(SecretBox.NONCE_SIZE)
    raw = json.dumps(data).encode()
    ciphertext = box.encrypt(raw, nonce).ciphertext
    encrypted = nonce + ciphertext
    decoded_data = base64.b64encode(encrypted).decode()
    return decoded_data, SecretBox.NONCE_SIZE


def decrypt_data(password: str, data: str) -> Any:
    key = derive_key(password)
    box = SecretBox(key)
    encrypted = base64.b64decode(data.encode())
    nonce = encrypted[: SecretBox.NONCE_SIZE]
    ciphertext = encrypted[SecretBox.NONCE_SIZE :]
    raw = box.decrypt(ciphertext, nonce)
    return json.loads(raw.decode())
