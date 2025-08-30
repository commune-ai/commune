import os
from typing import Optional, Dict, Any, List, Tuple, Union
import bittensor as bt

class Wallet:
    """
    A Python class that represents a Bittensor wallet.
    This class wraps the Rust implementation of the wallet functionality.
    """
    
    def __init__(self, name: Optional[str] = None, hotkey: Optional[str] = None, path: Optional[str] = None, config: Optional[Any] = None):
        """
        Initialize a wallet with the specified parameters.
        
        Args:
            name (str, optional): Name of the wallet. Defaults to environment variable BT_WALLET_NAME or 'default'.
            hotkey (str, optional): Name of the hotkey. Defaults to environment variable BT_WALLET_HOTKEY or 'default'.
            path (str, optional): Path to wallet files. Defaults to environment variable BT_WALLET_PATH or '~/.bittensor/wallets/'.
            config (Any, optional): Configuration object with wallet settings.
        """
        self._wallet = bt.Wallet(name, hotkey, path, config)
    
    def __str__(self) -> str:
        """
        Return a string representation of the wallet.
        
        Returns:
            str: String representation of the wallet.
        """
        return str(self._wallet)


    
    def create_if_non_existent(self, coldkey_use_password: bool = True, hotkey_use_password: bool = False, 
                              save_coldkey_to_env: bool = False, save_hotkey_to_env: bool = False,
                              coldkey_password: Optional[str] = None, hotkey_password: Optional[str] = None,
                              overwrite: bool = False, suppress: bool = False) -> 'Wallet':
        """
        Checks for existing coldkeypub and hotkeys, and creates them if non-existent.
        
        Args:
            coldkey_use_password (bool): Whether to use a password for coldkey. Defaults to True.
            hotkey_use_password (bool): Whether to use a password for hotkey. Defaults to False.
            save_coldkey_to_env (bool): Whether to save a coldkey password to local env. Defaults to False.
            save_hotkey_to_env (bool): Whether to save a hotkey password to local env. Defaults to False.
            coldkey_password (Optional[str]): Coldkey password for encryption. Defaults to None.
            hotkey_password (Optional[str]): Hotkey password for encryption. Defaults to None.
            overwrite (bool): Whether to overwrite an existing keys. Defaults to False.
            suppress (bool): If True, suppresses the display of the keys mnemonic message. Defaults to False.
            
        Returns:
            Wallet: Wallet instance with created keys.
        """
        result = self._wallet.create_if_non_existent(
            coldkey_use_password, hotkey_use_password, 
            save_coldkey_to_env, save_hotkey_to_env,
            coldkey_password, hotkey_password,
            overwrite, suppress
        )
        return Wallet()
    
    def create(self, coldkey_use_password: bool = True, hotkey_use_password: bool = False, 
              save_coldkey_to_env: bool = False, save_hotkey_to_env: bool = False,
              coldkey_password: Optional[str] = None, hotkey_password: Optional[str] = None,
              overwrite: bool = False, suppress: bool = False) -> 'Wallet':
        """
        Creates new coldkey and hotkey for this wallet.
        
        Args:
            coldkey_use_password (bool): Whether to use a password for coldkey. Defaults to True.
            hotkey_use_password (bool): Whether to use a password for hotkey. Defaults to False.
            save_coldkey_to_env (bool): Whether to save a coldkey password to local env. Defaults to False.
            save_hotkey_to_env (bool): Whether to save a hotkey password to local env. Defaults to False.
            coldkey_password (Optional[str]): Coldkey password for encryption. Defaults to None.
            hotkey_password (Optional[str]): Hotkey password for encryption. Defaults to None.
            overwrite (bool): Whether to overwrite an existing keys. Defaults to False.
            suppress (bool): If True, suppresses the display of the keys mnemonic message. Defaults to False.
            
        Returns:
            Wallet: Wallet instance with created keys.
        """
        result = self._wallet.create(
            coldkey_use_password, hotkey_use_password, 
            save_coldkey_to_env, save_hotkey_to_env,
            coldkey_password, hotkey_password,
            overwrite, suppress
        )
        return Wallet()
    
    def recreate(self, coldkey_use_password: bool = True, hotkey_use_password: bool = False, 
                save_coldkey_to_env: bool = False, save_hotkey_to_env: bool = False,
                coldkey_password: Optional[str] = None, hotkey_password: Optional[str] = None,
                overwrite: bool = False, suppress: bool = False) -> 'Wallet':
        """
        Recreates coldkey and hotkey for this wallet.
        
        Args:
            coldkey_use_password (bool): Whether to use a password for coldkey. Defaults to True.
            hotkey_use_password (bool): Whether to use a password for hotkey. Defaults to False.
            save_coldkey_to_env (bool): Whether to save a coldkey password to local env. Defaults to False.
            save_hotkey_to_env (bool): Whether to save a hotkey password to local env. Defaults to False.
            coldkey_password (Optional[str]): Coldkey password for encryption. Defaults to None.
            hotkey_password (Optional[str]): Hotkey password for encryption. Defaults to None.
            overwrite (bool): Whether to overwrite an existing keys. Defaults to False.
            suppress (bool): If True, suppresses the display of the keys mnemonic message. Defaults to False.
            
        Returns:
            Wallet: Wallet instance with recreated keys.
        """
        result = self._wallet.recreate(
            coldkey_use_password, hotkey_use_password, 
            save_coldkey_to_env, save_hotkey_to_env,
            coldkey_password, hotkey_password,
            overwrite, suppress
        )
        return Wallet()

    def list_keys(self) -> Dict[str, Any]:
        """
        Lists all keys in the wallet.
        
        Returns:
            Dict[str, Any]: Dictionary containing the keys and their details.
        """
        return self._wallet.list_keys()
    
    def get_coldkey(self, password: Optional[str] = None) -> 'Keypair':
        """
        Returns the coldkey from path, decrypts data if the file is encrypted.
        
        Args:
            password (Optional[str]): Password to decrypt the coldkey. Defaults to None.
            
        Returns:
            Keypair: The coldkey keypair.
        """
        return Keypair(self._wallet.get_coldkey(password))
    
    def get_coldkeypub(self, password: Optional[str] = None) -> 'Keypair':
        """
        Returns the coldkeypub from path, decrypts data if the file is encrypted.
        
        Args:
            password (Optional[str]): Password to decrypt the coldkeypub. Defaults to None.
            
        Returns:
            Keypair: The coldkeypub keypair.
        """
        return Keypair(self._wallet.get_coldkeypub(password))
    
    def get_hotkey(self, password: Optional[str] = None) -> 'Keypair':
        """
        Returns the hotkey from path, decrypts data if the file is encrypted.
        
        Args:
            password (Optional[str]): Password to decrypt the hotkey. Defaults to None.
            
        Returns:
            Keypair: The hotkey keypair.
        """
        return Keypair(self._wallet.get_hotkey(password))
    
    def set_coldkey(self, keypair: 'Keypair', encrypt: bool = True, overwrite: bool = False,
                   save_coldkey_to_env: bool = False, coldkey_password: Optional[str] = None) -> None:
        """
        Sets the coldkey at path.
        
        Args:
            keypair (Keypair): Keypair to set as coldkey.
            encrypt (bool): Whether to encrypt the coldkey. Defaults to True.
            overwrite (bool): Whether to overwrite an existing coldkey. Defaults to False.
            save_coldkey_to_env (bool): Whether to save the coldkey password to environment. Defaults to False.
            coldkey_password (Optional[str]): Password to encrypt the coldkey. Defaults to None.
        """
        self._wallet.set_coldkey(keypair._keypair, encrypt, overwrite, save_coldkey_to_env, coldkey_password)
    
    def set_coldkeypub(self, keypair: 'Keypair', encrypt: bool = False, overwrite: bool = False) -> None:
        """
        Sets the coldkeypub at path.
        
        Args:
            keypair (Keypair): Keypair to set as coldkeypub.
            encrypt (bool): Whether to encrypt the coldkeypub. Defaults to False.
            overwrite (bool): Whether to overwrite an existing coldkeypub. Defaults to False.
        """
        self._wallet.set_coldkeypub(keypair._keypair, encrypt, overwrite)
    
    def set_hotkey(self, keypair: 'Keypair', encrypt: bool = False, overwrite: bool = False,
                  save_hotkey_to_env: bool = False, hotkey_password: Optional[str] = None) -> None:
        """
        Sets the hotkey at path.
        
        Args:
            keypair (Keypair): Keypair to set as hotkey.
            encrypt (bool): Whether to encrypt the hotkey. Defaults to False.
            overwrite (bool): Whether to overwrite an existing hotkey. Defaults to False.
            save_hotkey_to_env (bool): Whether to save the hotkey password to environment. Defaults to False.
            hotkey_password (Optional[str]): Password to encrypt the hotkey. Defaults to None.
        """
        self._wallet.set_hotkey(keypair._keypair, encrypt, overwrite, save_hotkey_to_env, hotkey_password)
    
    @property
    def coldkey(self) -> 'Keypair':
        """
        Returns the coldkey property.
        
        Returns:
            Keypair: The coldkey keypair.
        """
        return Keypair(self._wallet.coldkey)
    
    @property
    def coldkeypub(self) -> 'Keypair':
        """
        Returns the coldkeypub property.
        
        Returns:
            Keypair: The coldkeypub keypair.
        """
        return Keypair(self._wallet.coldkeypub)
    
    @property
    def hotkey(self) -> 'Keypair':
        """
        Returns the hotkey property.
        
        Returns:
            Keypair: The hotkey keypair.
        """
        return Keypair(self._wallet.hotkey)
    
    @property
    def coldkey_file(self) -> 'Keyfile':
        """
        Returns the coldkey file.
        
        Returns:
            Keyfile: The coldkey file.
        """
        return Keyfile(self._wallet.coldkey_file)
    
    @property
    def coldkeypub_file(self) -> 'Keyfile':
        """
        Returns the coldkeypub file.
        
        Returns:
            Keyfile: The coldkeypub file.
        """
        return Keyfile(self._wallet.coldkeypub_file)
    
    @property
    def hotkey_file(self) -> 'Keyfile':
        """
        Returns the hotkey file.
        
        Returns:
            Keyfile: The hotkey file.
        """
        return Keyfile(self._wallet.hotkey_file)
    
    @property
    def name(self) -> str:
        """
        Returns the wallet name.
        
        Returns:
            str: The wallet name.
        """
        return self._wallet.name
    
    @property
    def path(self) -> str:
        """
        Returns the wallet path.
        
        Returns:
            str: The wallet path.
        """
        return self._wallet.path
    
    @property
    def hotkey_str(self) -> str:
        """
        Returns the hotkey string.
        
        Returns:
            str: The hotkey string.
        """
        return self._wallet.hotkey_str
    
    def create_coldkey_from_uri(self, uri: str, use_password: bool = True, overwrite: bool = False,
                               suppress: bool = False, save_coldkey_to_env: bool = False,
                               coldkey_password: Optional[str] = None) -> None:
        """
        Creates a coldkey from a URI.
        
        Args:
            uri (str): URI to create coldkey from.
            use_password (bool): Whether to use a password for coldkey. Defaults to True.
            overwrite (bool): Whether to overwrite an existing coldkey. Defaults to False.
            suppress (bool): If True, suppresses the display of the keys mnemonic message. Defaults to False.
            save_coldkey_to_env (bool): Whether to save the coldkey password to environment. Defaults to False.
            coldkey_password (Optional[str]): Password to encrypt the coldkey. Defaults to None.
        """
        self._wallet.create_coldkey_from_uri(uri, use_password, overwrite, suppress, save_coldkey_to_env, coldkey_password)
    
    def create_hotkey_from_uri(self, uri: str, use_password: bool = False, overwrite: bool = False,
                             suppress: bool = False, save_hotkey_to_env: bool = False,
                             hotkey_password: Optional[str] = None) -> None:
        """
        Creates a hotkey from a URI.
        
        Args:
            uri (str): URI to create hotkey from.
            use_password (bool): Whether to use a password for hotkey. Defaults to False.
            overwrite (bool): Whether to overwrite an existing hotkey. Defaults to False.
            suppress (bool): If True, suppresses the display of the keys mnemonic message. Defaults to False.
            save_hotkey_to_env (bool): Whether to save the hotkey password to environment. Defaults to False.
            hotkey_password (Optional[str]): Password to encrypt the hotkey. Defaults to None.
        """
        self._wallet.create_hotkey_from_uri(uri, use_password, overwrite, suppress, save_hotkey_to_env, hotkey_password)
    
    def unlock_coldkey(self) -> 'Keypair':
        """
        Unlocks the coldkey.
        
        Returns:
            Keypair: The unlocked coldkey keypair.
        """
        return Keypair(self._wallet.unlock_coldkey())
    
    def unlock_coldkeypub(self) -> 'Keypair':
        """
        Unlocks the coldkeypub.
        
        Returns:
            Keypair: The unlocked coldkeypub keypair.
        """
        return Keypair(self._wallet.unlock_coldkeypub())
    
    def unlock_hotkey(self) -> 'Keypair':
        """
        Unlocks the hotkey.
        
        Returns:
            Keypair: The unlocked hotkey keypair.
        """
        return Keypair(self._wallet.unlock_hotkey())
    
    def create_new_coldkey(self, n_words: int = 12, use_password: bool = True, overwrite: bool = False,
                          suppress: bool = False, save_coldkey_to_env: bool = False,
                          coldkey_password: Optional[str] = None) -> 'Wallet':
        """
        Creates a new coldkey.
        
        Args:
            n_words (int): Number of mnemonic words. Defaults to 12.
            use_password (bool): Whether to use a password for coldkey. Defaults to True.
            overwrite (bool): Whether to overwrite an existing coldkey. Defaults to False.
            suppress (bool): If True, suppresses the display of the keys mnemonic message. Defaults to False.
            save_coldkey_to_env (bool): Whether to save the coldkey password to environment. Defaults to False.
            coldkey_password (Optional[str]): Password to encrypt the coldkey. Defaults to None.
            
        Returns:
            Wallet: Wallet instance with new coldkey.
        """
        result = self._wallet.create_new_coldkey(n_words, use_password, overwrite, suppress, save_coldkey_to_env, coldkey_password)
        return Wallet()
    
    def create_new_hotkey(self, n_words: int = 12, use_password: bool = False, overwrite: bool = False,
                        suppress: bool = False, save_hotkey_to_env: bool = False,
                        hotkey_password: Optional[str] = None) -> 'Wallet':
        """
        Creates a new hotkey.
        
        Args:
            n_words (int): Number of mnemonic words. Defaults to 12.
            use_password (bool): Whether to use a password for hotkey. Defaults to False.
            overwrite (bool): Whether to overwrite an existing hotkey. Defaults to False.
            suppress (bool): If True, suppresses the display of the keys mnemonic message. Defaults to False.
            save_hotkey_to_env (bool): Whether to save the hotkey password to environment. Defaults to False.
            hotkey_password (Optional[str]): Password to encrypt the hotkey. Defaults to None.
            
        Returns:
            Wallet: Wallet instance with new hotkey.
        """
        result = self._wallet.create_new_hotkey(n_words, use_password, overwrite, suppress, save_hotkey_to_env, hotkey_password)
        return Wallet()
    
    def regenerate_coldkey(self, mnemonic: Optional[str] = None, seed: Optional[str] = None,
                         json: Optional[Tuple[str, str]] = None, use_password: bool = True,
                         overwrite: bool = False, suppress: bool = False,
                         save_coldkey_to_env: bool = False, coldkey_password: Optional[str] = None) -> None:
        """
        Regenerates coldkey from mnemonic, seed, or json.
        
        Args:
            mnemonic (Optional[str]): Mnemonic to regenerate coldkey from. Defaults to None.
            seed (Optional[str]): Seed to regenerate coldkey from. Defaults to None.
            json (Optional[Tuple[str, str]]): JSON and password to regenerate coldkey from. Defaults to None.
            use_password (bool): Whether to use a password for coldkey. Defaults to True.
            overwrite (bool): Whether to overwrite an existing coldkey. Defaults to False.
            suppress (bool): If True, suppresses the display of the keys mnemonic message. Defaults to False.
            save_coldkey_to_env (bool): Whether to save the coldkey password to environment. Defaults to False.
            coldkey_password (Optional[str]): Password to encrypt the coldkey. Defaults to None.
        """
        self._wallet.regenerate_coldkey(mnemonic, seed, json, use_password, overwrite, suppress, save_coldkey_to_env, coldkey_password)
    
    def regenerate_coldkeypub(self, ss58_address: Optional[str] = None, public_key: Optional[str] = None,
                           overwrite: bool = False) -> None:
        """
        Regenerates coldkeypub from ss58_address or public_key.
        
        Args:
            ss58_address (Optional[str]): SS58 address to regenerate coldkeypub from. Defaults to None.
            public_key (Optional[str]): Public key to regenerate coldkeypub from. Defaults to None.
            overwrite (bool): Whether to overwrite an existing coldkeypub. Defaults to False.
        """
        self._wallet.regenerate_coldkeypub(ss58_address, public_key, overwrite)
    
    def regenerate_hotkey(self, mnemonic: Optional[str] = None, seed: Optional[str] = None,
                       json: Optional[Tuple[str, str]] = None, use_password: bool = False,
                       overwrite: bool = False, suppress: bool = False,
                       save_hotkey_to_env: bool = False, hotkey_password: Optional[str] = None) -> None:
        """
        Regenerates hotkey from mnemonic, seed, or json.
        
        Args:
            mnemonic (Optional[str]): Mnemonic to regenerate hotkey from. Defaults to None.
            seed (Optional[str]): Seed to regenerate hotkey from. Defaults to None.
            json (Optional[Tuple[str, str]]): JSON and password to regenerate hotkey from. Defaults to None.
            use_password (bool): Whether to use a password for hotkey. Defaults to False.
            overwrite (bool): Whether to overwrite an existing hotkey. Defaults to False.
            suppress (bool): If True, suppresses the display of the keys mnemonic message. Defaults to False.
            save_hotkey_to_env (bool): Whether to save the hotkey password to environment. Defaults to False.
            hotkey_password (Optional[str]): Password to encrypt the hotkey. Defaults to None.
        """
        self._wallet.regenerate_hotkey(mnemonic, seed, json, use_password, overwrite, suppress, save_hotkey_to_env, hotkey_password)

