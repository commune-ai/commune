import os
import base64
import json
import stat
import getpass
from typing import Optional
from pathlib import Path

from ansible_vault import Vault
from cryptography.exceptions import InvalidSignature, InvalidKey
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from password_strength import PasswordPolicy
from substrateinterface.utils.ss58 import ss58_encode
from termcolor import colored


import os
import sys
from types import SimpleNamespace
from typing import Optional, Union

import bittensor
from substrateinterface import Keypair
from termcolor import colored


import os
import base64
import json
import stat
import getpass
from typing import Optional
from pathlib import Path


from ansible_vault import Vault
from cryptography.exceptions import InvalidSignature, InvalidKey
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from password_strength import PasswordPolicy
from substrateinterface.utils.ss58 import ss58_encode
from termcolor import colored

# Substrate ss58_format
__ss58_format__ = 42
# Wallet ss58 address length
__ss58_address_length__ = 48

class KeyFileError(Exception):
    """ Error thrown when the keyfile is corrupt, non-writable, nno-readable or the password used to decrypt is invalid.
    """


class SubstrateAccount(KeyPair):
    def __init__(self, *args, **kwargs):
        KeyPair.__init__(self, *args, **kwargs)


if __name__ == '__main__':
    module = SubstrateAccount()
    st.write(module)
