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




class SubstrateAccountModule(Keypair):
    """
    Subsrate wallet maintenance class. Each wallet contains a key. 
    The key is the user's primary key for holding stake in their wallet.
   
       """
    module_dir = os.path.dir(__file__) 
    def __init__( 
        self,
        account:str = None,
    ):
        r""" Init bittensor wallet object containing a  key.
            Args:
                account (required=True, default='~/.bittensor/wallets/'):
                    The path to your bittensor wallets
        """


        self.set_account(account)


    def __str__(self):
        return "Wallet ({}, {}, {})".format(self.name, self.path)
    
    def __repr__(self):
        return self.__str__()

    def create_if_non_existent( self, key_use_password:bool = True) -> 'Wallet':
        """ Checks for existing keypub and hotkeys and creates them if non-existent.
        """
        return self.create(key_use_password, hotkey_use_password)

    def create (self, key_use_password:bool = True ) -> 'Wallet':
        """ Checks for existing keypub and hotkeys and creates them if non-existent.
        """
        # ---- Setup Wallet. ----
        if not self.key_file.exists_on_device() and not self.keypub_file.exists_on_device():
            self.create_new_key( n_words = 12, use_password = key_use_password )
        return self

    def recreate (self, key_use_password:bool = True ) -> 'Wallet':
        """ Checks for existing keypub and hotkeys and creates them if non-existent.
        """
        # ---- Setup Wallet. ----
        self.create_new_key( n_words = 12, use_password = key_use_password )
        return self

    @property
    def key_file(self) -> 'AccountFileModule':
        wallet_path = os.path.expanduser(os.path.join(self.path, self.name))
        key_path = os.path.join(wallet_path, "key")
        return AccountFileModule( path = key_path )

    @property
    def wallet_path(self):
        return os.path.expanduser(os.path.join(self.path, self.name))

    @property
    def keypub_file(self) -> 'AccountFileModule':
        keypub_path = os.path.join(self.wallet_path, "keypub.txt")
        return AccountFileModule( path = keypub_path )

    def set_key(self, keypair: 'Keypair', encrypt: bool = False, overwrite: bool = False,  password:Optional[str]=None) -> 'AccountFileModule':
        self._key = keypair
        self.key_file.set_keypair( self._key, encrypt = encrypt, overwrite = overwrite , password=password )

    def set_keypub(self, keypair: 'Keypair', encrypt: bool = False, overwrite: bool = False,  password:Optional[str]=None) -> 'AccountFileModule':
        self._keypub = Keypair(ss58_address=keypair.ss58_address)
        self.keypub_file.set_keypair( self._keypub, encrypt = encrypt, overwrite = overwrite , password=password )

    def get_key(self, password: str = None ) -> 'Keypair':
        self.key_file.get_keypair( password = password )

    def get_keypub(self, password: str = None ) -> 'Keypair':
        self.keypub_file.get_keypair( password = password )

    @property
    def key(self) -> 'Keypair':
        r""" Loads the hotkey from wallet.path/wallet.name/key or raises an error.
            Returns:
                key (Keypair):
                    colkey loaded from config arguments.
            Raises:
                KeyFileError: Raised if the file is corrupt of non-existent.
                CryptoKeyError: Raised if the user enters an incorrec password for an encrypted keyfile.
        """
        if self._key == None:
            self._key = self.key_file.keypair
        return self._key

    @property
    def keypub(self) -> 'Keypair':
        r""" Loads the keypub from wallet.path/wallet.name/keypub.txt or raises an error.
            Returns:
                keypub (Keypair):
                    colkeypub loaded from config arguments.
            Raises:
                KeyFileError: Raised if the file is corrupt of non-existent.
                CryptoKeyError: Raised if the user enters an incorrect password for an encrypted keyfile.
        """
        if self._keypub == None:
            self._keypub = self.keypub_file.keypair
        return self._keypub
            
    def create_key_from_uri(self, uri:str, use_password: bool = True, overwrite:bool = False) -> 'Wallet':
        """ Creates key from suri string, optionally encrypts it with the user's inputed password.
            Args:
                uri: (str, required):
                    URI string to use i.e. /Alice or /Bob
                use_password (bool, optional):
                    Is the created key password protected.
                overwrite (bool, optional): 
                    Will this operation overwrite the key under the same path <wallet path>/<wallet name>/key
            Returns:
                wallet (Wallet):
                    this object with newly created key.
        """
        keypair = Keypair.create_from_uri( uri )
        self.display_mnemonic_msg( keypair, "key" )
        self.set_key( keypair, encrypt = use_password, overwrite = overwrite)
        self.set_keypub( keypair, overwrite = overwrite)
        return self

    def new_key( self, n_words:int = 12, use_password: bool = True, overwrite:bool = False) -> 'Wallet':  
        """ Creates a new key, optionally encrypts it with the user's inputed password and saves to disk.
            Args:
                n_words: (int, optional):
                    Number of mnemonic words to use.
                use_password (bool, optional):
                    Is the created key password protected.
                overwrite (bool, optional): 
                    Will this operation overwrite the key under the same path <wallet path>/<wallet name>/key
            Returns:
                wallet (Wallet):
                    this object with newly created key.
        """
        self.create_new_key( n_words, use_password, overwrite )

    @staticmethod
    def create_from_mnemonic(mnemonic:str):
        return Keypair.create_from_mnemonic(mnemonic)

    def regen_key( self, mnemonic: Optional[Union[list, str]]=None, seed: Optional[str]=None, use_password: bool = True,  overwrite:bool = False) -> 'Wallet':
        """ Regenerates the key from passed mnemonic, encrypts it with the user's password and save the file
            Args:
                mnemonic: (Union[list, str], optional):
                    Key mnemonic as list of words or string space separated words.
                seed: (str, optional):
                    Seed as hex string.
                use_password (bool, optional):
                    Is the created key password protected.
                overwrite (bool, optional): 
                    Will this operation overwrite the key under the same path <wallet path>/<wallet name>/key
            Returns:
                wallet (Wallet):
                    this object with newly created key.
        """
        self.regenerate_key(mnemonic, seed, use_password, overwrite)

    def regenerate_keypub( self, ss58_address: Optional[str] = None, public_key: Optional[Union[str, bytes]] = None, overwrite: bool = False ) -> 'Wallet':
        """ Regenerates the keypub from passed ss58_address or public_key and saves the file
               Requires either ss58_address or public_key to be passed.
            Args:
                ss58_address: (str, optional):
                    Address as ss58 string.
                public_key: (str | bytes, optional):
                    Public key as hex string or bytes.
                overwrite (bool, optional) (default: False):
                    Will this operation overwrite the keypub (if exists) under the same path <wallet path>/<wallet name>/keypub
            Returns:
                wallet (Wallet):
                    newly re-generated Wallet with keypub.
            
        """
        if ss58_address is None and public_key is None:
            raise ValueError("Either ss58_address or public_key must be passed")

        if not is_valid_address_or_public_key( ss58_address if ss58_address is not None else public_key ):
            raise ValueError(f"Invalid {'ss58_address' if ss58_address is not None else 'public_key'}") 

        keypair = Keypair(ss58_address=ss58_address, public_key=public_key, ss58_format=__ss58_format__)

        # No need to encrypt the public key
        self.set_keypub( keypair, overwrite = overwrite)

        return self

    # Short name for regenerate_keypub
    regen_keypub = regenerate_keypub

    def regenerate_key( self, mnemonic: Optional[Union[list, str]] = None, seed: Optional[str] = None, use_password: bool = True,  overwrite:bool = False) -> 'Wallet':
        """ Regenerates the key from passed mnemonic, encrypts it with the user's password and save the file
            Args:
                mnemonic: (Union[list, str], optional):
                    Key mnemonic as list of words or string space separated words.
                seed: (str, optional):
                    Seed as hex string.
                use_password (bool, optional):
                    Is the created key password protected.
                overwrite (bool, optional): 
                    Will this operation overwrite the key under the same path <wallet path>/<wallet name>/key
            Returns:
                wallet (Wallet):
                    this object with newly created key.
        """
        if mnemonic is None and seed is None:
            raise ValueError("Must pass either mnemonic or seed")
        if mnemonic is not None:
            if isinstance( mnemonic, str): mnemonic = mnemonic.split()
            if len(mnemonic) not in [12,15,18,21,24]:
                raise ValueError("Mnemonic has invalid size. This should be 12,15,18,21 or 24 words")
            keypair = Keypair.create_from_mnemonic(" ".join(mnemonic))   
            self.display_mnemonic_msg( keypair, "key" )
        else:
            # seed is not None
            keypair = Keypair.create_from_seed(seed)
            
        self.set_key( keypair, encrypt = use_password, overwrite = overwrite)
        self.set_keypub( keypair, overwrite = overwrite)
        return self 

    @staticmethod
    def display_mnemonic_msg( keypair : Keypair, key_type : str ):
        """ Displaying the mnemonic and warning message to keep mnemonic safe
        """
        mnemonic = keypair.mnemonic
        mnemonic_green = colored(mnemonic, 'green')
        print (colored("\nIMPORTANT: Store this mnemonic in a secure (preferable offline place), as anyone " \
                    "who has possesion of this mnemonic can use it to regenerate the key and access your tokens. \n", "red"))
        print ("The mnemonic to the new {} is:\n\n{}\n".format(key_type, mnemonic_green))
        print ("You can use the mnemonic to recreate the key in case it gets lost. The command to use to regenerate the key using this mnemonic is:")
        print("btcli regen_{} --mnemonic {}".format(key_type, mnemonic))
        print('')




    def set_account(self, account:str)
        '''sets the accoutn'''


        if isinstance(account, str)

            if os.path.exists(account):
                self.path = account
            if '::' in account:
                account_mode, account =  
            self.name = account
        else:
            raise NotImplementedError
            

    @property
    def default_path(self) -> str:
        return  f'{self.module_dir}/{self.name}'

    @property
    def path(self) -> str:
        if not hasattr(self, '_path'):
            self._path = self.default_path
        return self._path

    @path.setter
    def path(self, path:str) -> str:
        self._path = path
        return self._path

    def __str__(self):
        if not self.exists_on_device():
            return "AccountFileModule (empty, {})>".format( self.path )
        if self.is_encrypted():
            return "AccountFileModule (encrypted, {})>".format( self.path )
        else:
            return "AccountFileModule (decrypted, {})>".format( self.path )

    def __repr__(self):
        return self.__str__()

    @property
    def keypair( self ) -> 'Keypair':
        """ Returns the keypair from path, decrypts data if the file is encrypted.
            Args:
                password ( str, optional ):
                    Optional password used to decrypt file. If None, asks for user input.
            Returns:
                keypair (Keypair):
                    Keypair stored under path.
            Raises:
                KeyFileError:
                    Raised if the file does not exists, is not readable, writable 
                    corrupted, or if the password is incorrect.
        """
        return self.get_keypair()

    @property
    def data( self ) -> bytes:
        """ Returns keyfile data under path.
            Returns:
                keyfile_data (bytes):   
                    AccountFileModule data stored under path.
            Raises:
                KeyFileError:
                    Raised if the file does not exists, is not readable, or writable.
        """
        return self._read_keyfile_data_from_file()

    @property
    def keyfile_data( self ) -> bytes:
        """ Returns keyfile data under path.
            Returns:
                keyfile_data (bytes):   
                    AccountFileModule data stored under path.
            Raises:
                KeyFileError:
                    Raised if the file does not exists, is not readable, or writable.
        """
        return self._read_keyfile_data_from_file()

    def set_keypair ( self, keypair: 'Keypair', password:str = None, overwrite: bool = False):
        """ Writes the keypair to the file and optional encrypts data.
            Args:
                keypair (Keypair):
                    Keypair to store under path.
                password ( str, optional ):
                    Optional password used to encrypt file. If None, asks for user input.
                overwrite ( bool, optional, default = True ):
                    If True, forces overwrite of current file.
            Raises:
                KeyFileError:
                    Raised if the file does not exists, is not readable, or writable.
        """
        self.make_dirs()
        keyfile_data = self.serialized_keypair_to_keyfile_data( keypair )
        if pasword:
            keyfile_data = self.encrypt_keyfile_data( keyfile_data, password )
        self._write_keyfile_data_to_file( keyfile_data, overwrite = overwrite )

    def get_keypair(self, password: str = None) -> 'Keypair':
        """ Returns the keypair from path, decrypts data if the file is encrypted.
            Args:
                password ( str, optional ):
                    Optional password used to decrypt file. If None, asks for user input.
            Returns:
                keypair (Keypair):
                    Keypair stored under path.
            Raises:
                KeyFileError:
                    Raised if the file does not exists, is not readable, writable 
                    corrupted, or if the password is incorrect.
        """
        keyfile_data = self._read_keyfile_data_from_file()
        if self.keyfile_data_is_encrypted( keyfile_data=keyfile_data ):
            keyfile_data = self.decrypt_keyfile_data(keyfile_data=keyfile_data, password=password)
        return self.deserialize_keypair_from_keyfile_data( keyfile_data )

    def make_dirs( self ):
        """ Makes directories for path.
        """
        directory = os.path.dirname( self.path )
        if not os.path.exists( directory ):
            os.makedirs( directory ) 

    def exists_on_device( self ) -> bool:
        """ Returns true if the file exists on the device.
            Returns:
                on_device (bool):
                    True if the file is on device.
        """
        if not os.path.isfile( self.path ):
            return False
        return True

    def is_readable( self ) -> bool:
        """ Returns true if the file under path is readable.
            Returns:
                readable (bool):
                    True if the file is readable.
        """
        if not self.exists_on_device():
            return False
        if not os.access( self.path , os.R_OK ):
            return False
        return True

    def is_writable( self ) -> bool:
        """ Returns true if the file under path is writable.
            Returns:
                writable (bool):
                    True if the file is writable.
        """
        if os.access(self.path, os.W_OK):
            return True
        return False

    def is_encrypted ( self ) -> bool:
        """ Returns true if the file under path is encrypted.
            Returns:
                encrypted (bool):
                    True if the file is encrypted.
        """
        if not self.exists_on_device():
            return False
        if not self.is_readable():
            return False
        return self.keyfile_data_is_encrypted( self._read_keyfile_data_from_file() )

    def _may_overwrite ( self ) -> bool:
        choice = input("File {} already exists. Overwrite ? (y/N) ".format( self.path ))
        return choice == 'y'

    def encrypt( self, password: str = None):
        """ Encrypts file under path.
            Args:
                password: (str, optional):
                    Optional password for encryption. Otherwise asks for user input.
            Raises:
                KeyFileError:
                    Raised if the file does not exists, is not readable, writable.
        """
        if not self.exists_on_device():
            raise KeyFileError( "AccountFileModule at: {} is not a file".format( self.path ))
        if not self.is_readable():
            raise KeyFileError( "AccountFileModule at: {} is not readable".format( self.path ))
        if not self.is_writable():
            raise KeyFileError( "AccountFileModule at: {} is not writeable".format( self.path ) ) 
        keyfile_data = self._read_keyfile_data_from_file()
        if not self.keyfile_data_is_encrypted( keyfile_data ):
            as_keypair = self.deserialize_keypair_from_keyfile_data( keyfile_data )
            keyfile_data = self.serialized_keypair_to_keyfile_data( as_keypair )
            keyfile_data = self.encrypt_keyfile_data( keyfile_data, password )
        self._write_keyfile_data_to_file( keyfile_data, overwrite = True )

    def decrypt( self, password: str = None):
        """ Decrypts file under path.
            Args:
                password: (str, optional):
                    Optional password for decryption. Otherwise asks for user input.
            Raises:
                KeyFileError:
                    Raised if the file does not exists, is not readable, writable 
                    corrupted, or if the password is incorrect.
        """
        if not self.exists_on_device():
            raise KeyFileError( "AccountFileModule at: {} is not a file".format( self.path ))
        if not self.is_readable():
            raise KeyFileError( "AccountFileModule at: {} is not readable".format( self.path ))
        if not self.is_writable():
            raise KeyFileError( "No write access for {}".format( self.path ) ) 
        keyfile_data = self._read_keyfile_data_from_file()
        if self.keyfile_data_is_encrypted( keyfile_data ):
            keyfile_data = self.decrypt_keyfile_data(keyfile_data, password)
        as_keypair = self.deserialize_keypair_from_keyfile_data( keyfile_data )
        keyfile_data = self.serialized_keypair_to_keyfile_data( as_keypair )
        self._write_keyfile_data_to_file( keyfile_data, overwrite = True )

    def _read_keyfile_data_from_file ( self ) -> bytes:
        """ Reads keyfile data from path.
            Returns:
                keyfile_data: (bytes, required):
                    AccountFileModule data sotred under path.
            Raises:
                KeyFileError:
                    Raised if the file does not exists or is not readable.
        """
        if not self.exists_on_device():
            raise KeyFileError( "AccountFileModule at: {} is not a file".format( self.path ))
        if not self.is_readable():
            raise KeyFileError( "AccountFileModule at: {} is not readable".format( self.path ))
        with open( self.path , 'rb') as file:
            data = file.read()
        return data

    def _write_keyfile_data_to_file ( self, keyfile_data:bytes, overwrite: bool = False ):
        """ Writes the keyfile data to path, if overwrite is true, forces operation without asking.
            Args:
                keyfile_data: (bytes, required):
                    Byte data to store under path.
                overwrite (bool, optional):
                    If True, overwrites data without asking for overwrite permissions from the user.
            Raises:
                KeyFileError:
                    Raised if the file is not writable or the user returns No to overwrite prompt.
        """
        # Check overwrite.
        if self.exists_on_device() and not overwrite:
            if not self._may_overwrite():
                raise KeyFileError( "AccountFileModule at: {} is not writeable".format( self.path ) ) 
        with open(self.path, "wb") as keyfile:
            keyfile.write( keyfile_data )
        # Set file permissions.
        os.chmod(self.path, stat.S_IRUSR | stat.S_IWUSR)

    @staticmethod
    def serialized_keypair_to_keyfile_data( keypair: 'Keypair' ):
        """ Serializes keypair object into keyfile data.
            Args:
                password ( str, required ):
                    password to verify.
            Returns:
                valid ( bool ):
                    True if the password meets validity requirements.
        """
        json_data = {
            'accountId': "0x" + keypair.public_key.hex() if keypair.public_key != None else None,
            'publicKey': "0x" + keypair.public_key.hex()  if keypair.public_key != None else None,
            'secretPhrase': keypair.mnemonic if keypair.mnemonic != None else None,
            'secretSeed': "0x" + keypair.seed_hex if keypair.seed_hex != None else None,
            'ss58Address': keypair.ss58_address if keypair.ss58_address != None else None
        }
        data = json.dumps( json_data ).encode()
        return data
    
    @staticmethod
    def deserialize_keypair_from_keyfile_data( keyfile_data:bytes ) -> 'Keypair':
        """ Deserializes Keypair object from passed keyfile data.
            Args:
                keyfile_data ( bytest, required ):
                    AccountFileModule data as bytes to be loaded.
            Returns:
                keypair (Keypair):
                    Keypair loaded from bytes.
            Raises:
                KeyFileError:
                    Raised if the passed bytest cannot construct a keypair object.
        """
        # Decode from json.
        keyfile_data = keyfile_data.decode()
        try:
            keyfile_dict = dict(json.loads( keyfile_data ))
        except:
            string_value = str(keyfile_data)
            if string_value[:2] == "0x":
                string_value = ss58_encode( string_value )
                keyfile_dict = {
                    'accountId': None,
                    'publicKey': None,
                    'secretPhrase': None,
                    'secretSeed': None,
                    'ss58Address': string_value
                }
            else:
                raise KeyFileError('Keypair could not be created from keyfile data: {}'.format( string_value ))

        if "secretSeed" in keyfile_dict and keyfile_dict['secretSeed'] != None:
            return Keypair.create_from_seed(keyfile_dict['secretSeed'])

        if "secretPhrase" in keyfile_dict and keyfile_dict['secretPhrase'] != None:
            return Keypair.create_from_mnemonic(mnemonic=keyfile_dict['secretPhrase'])

        if "ss58Address" in keyfile_dict and keyfile_dict['ss58Address'] != None:
            return Keypair( ss58_address = keyfile_dict['ss58Address'] )

        else:
            raise KeyFileError('Keypair could not be created from keyfile data: {}'.format( keyfile_dict ))
    

    
    @staticmethod
    def keyfile_data_is_encrypted_ansible( keyfile_data:bytes ) -> bool:
        """ Returns true if the keyfile data is ansible encrypted.
            Args:
                keyfile_data ( bytes, required ):
                    Bytes to validate
            Returns:
                is_ansible (bool):
                    True if data is ansible encrypted.
        """
        return keyfile_data[:14] == b'$ANSIBLE_VAULT'
    
    @staticmethod
    def keyfile_data_is_encrypted_legacy( keyfile_data:bytes ) -> bool:
        """ Returns true if the keyfile data is legacy encrypted.
            Args:
                keyfile_data ( bytes, required ):
                    Bytes to validate
            Returns:
                is_legacy (bool):
                    True if data is legacy encrypted.
        """
        return keyfile_data[:6] == b"gAAAAA"
    
    @staticmethod
    def keyfile_data_is_encrypted( keyfile_data:bytes ) -> bool:
        """ Returns true if the keyfile data is encrypted.
            Args:
                keyfile_data ( bytes, required ):
                    Bytes to validate
            Returns:
                is_encrypted (bool):
                    True if data is encrypted.
        """
        return SubstrateAccountModule.keyfile_data_is_encrypted_ansible( keyfile_data ) or keyfile_data_is_encrypted_legacy( keyfile_data )

    @staticmethod
    def encrypt_keyfile_data ( keyfile_data:bytes, password: str = None ) -> bytes:
        """ Encrypts passed keyfile data using ansible vault.
            Args:
                keyfile_data ( bytes, required ):
                    Bytes to validate
                password ( bool, optional ):
                    It set, uses this password to encrypt data.
            Returns:
                encrytped_data (bytes):
                    Ansible encrypted data.
        """
        with console.status(":locked_with_key: Encrypting key..."):
            vault = Vault( password )
        return vault.vault.encrypt ( keyfile_data )

    @staticmethod
    def decrypt_keyfile_data(keyfile_data: bytes, password: str = None) -> bytes:
        """ Decrypts passed keyfile data using ansible vault.
            Args:
                keyfile_data ( bytes, required ):
                    Bytes to validate
                password ( bool, optional ):
                    It set, uses this password to decrypt data.
            Returns:
                decrypted_data (bytes):
                    Decrypted data.
            Raises:
                KeyFileError:
                    Raised if the file is corrupted or if the password is incorrect.
        """
        try:
            if keyfile_data_is_encrypted_ansible( keyfile_data ):
                vault = Vault( password )
                decrypted_keyfile_data = vault.load( keyfile_data )
            # Legacy decrypt.
            elif keyfile_data_is_encrypted_legacy( keyfile_data ):
                __SALT = b"Iguesscyborgslikemyselfhaveatendencytobeparanoidaboutourorigins"
                kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), salt=__SALT, length=32, iterations=10000000, backend=default_backend())
                key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
                cipher_suite = Fernet(key)
                decrypted_keyfile_data = cipher_suite.decrypt( keyfile_data )   
            # Unknown.
            else: 
                raise KeyFileError( "AccountFileModule data: {} is corrupt".format( keyfile_data ))

        except (InvalidSignature, InvalidKey, InvalidToken):
            raise KeyFileError('Invalid password')

        if not isinstance(decrypted_keyfile_data, bytes):
            decrypted_keyfile_data = json.dumps( decrypted_keyfile_data ).encode()
        return decrypted_keyfile_data

    @staticmethod
    def is_valid_ss58_address( address: str, ss58_format = __ss58_format__ ) -> bool:
        """
        Checks if the given address is a valid ss58 address.

        Args:
            address(str): The address to check.

        Returns:
            True if the address is a valid ss58 address for Bittensor, False otherwise.
        """
        try:
            return ss58.is_valid_ss58_address( address, valid_ss58_format=ss58_format )
        except (IndexError):
            return False

    @staticmethod
    def is_valid_ed25519_pubkey( public_key: Union[str, bytes] , ss58_format=__ss58_format__ ) -> bool:
        """
        Checks if the given public_key is a valid ed25519 key.

        Args:
            public_key(Union[str, bytes]): The public_key to check.

        Returns:    
            True if the public_key is a valid ed25519 key, False otherwise.
        
        """
        try:
            if isinstance( public_key, str ):
                if len(public_key) != 64 and len(public_key) != 66:
                    raise ValueError( "a public_key should be 64 or 66 characters" )
            elif isinstance( public_key, bytes ):
                if len(public_key) != 32:
                    raise ValueError( "a public_key should be 32 bytes" )
            else:
                raise ValueError( "public_key must be a string or bytes" )

            keypair = Keypair(
                public_key=public_key,
                ss58_format=ss58_format
            )

            ss58_addr = keypair.ss58_address
            return ss58_addr is not None

        except (ValueError, IndexError):
            return False

    @staticmethod
    def is_valid_address_or_public_key( address: Union[str, bytes] ) -> bool:
        """
        Checks if the given address is a valid destination address.

        Args:
            address(Union[str, bytes]): The address to check.

        Returns:
            True if the address is a valid destination address, False otherwise.
        """
        if isinstance( address, str ):
            # Check if ed25519
            if address.startswith('0x'):
                return self.is_valid_ed25519_pubkey( address )
            else:
                # Assume ss58 address
                return self.is_valid_ss58_address( address )
        elif isinstance( address, bytes ):
            # Check if ed25519
            return self.is_valid_ed25519_pubkey( address )
        else:
            # Invalid address type
            return False


