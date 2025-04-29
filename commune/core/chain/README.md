# Commune Chain Module

This module provides a Python interface for interacting with Substrate-based blockchains. It leverages the `scalecodec` library for SCALE encoding/decoding and offers a high-level API for common blockchain operations such as querying storage, submitting extrinsics, and managing keys.

## Features

*   **Substrate Interface:** Provides a `SubstrateInterface` class for interacting with Substrate nodes via HTTP or WebSocket.
*   **Key Management:** Includes `Keypair` and related classes for generating, managing, and deriving cryptographic keys.
*   **SCALE Codec Support:** Uses the `scalecodec` library for encoding and decoding data structures according to the Substrate SCALE codec specification.
*   **Metadata Handling:**  Fetches and parses chain metadata for runtime information, storage layouts, and call definitions.
*   **Extrinsic Submission:**  Supports composing, signing, and submitting extrinsics with options for waiting for inclusion and finalization.
*   **Storage Queries:**  Provides methods for querying chain storage, including support for mapped storage items and batch queries.
*   **Multisig Support:**  Offers functionality for creating and managing multisignature accounts and extrinsics.
*   **Event Handling:**  Retrieves and processes events associated with blocks and extrinsics.
*   **Extension Mechanism:** Allows extending the core functionality with custom extensions.
*   **Thread Pool Execution:** Uses a thread pool to execute batch requests to the Substrate node.
*   **Automatic Type Registry Reloading:** Reloads type registry and preset used to instantiate the SubtrateInterface object. Useful to periodically apply changes in type definitions when a runtime upgrade occurred.

## Installation

```bash
git clone https://github.com/commune-ai/commune.git 
pip install -e ./commune
```

## Usage

### Basic Example

```python
from commune import Chain, Keypair

# Connect to a Substrate node
chain = Chain(network='main')

# Generate a keypair
keypair = Keypair.create_from_uri('//Alice')

# Query the balance of an account
balance = chain.balance(keypair.ss58_address)
print(f"Balance of Alice: {balance}")

# Transfer some tokens
recipient = '5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY'  # Bob's address
amount = 1000
receipt = chain.transfer(keypair, recipient, amount)

if receipt['success']:
    print(f"Transfer successful! Extrinsic hash: {receipt['tx_hash']}")
else:
    print(f"Transfer failed: {receipt['error']}")
```

### Key Concepts

*   **`SubstrateInterface`:** The main class for interacting with a Substrate node. It handles the connection, metadata retrieval, and RPC requests.
*   **`Keypair`:** Represents a cryptographic keypair used for signing transactions.  Can be created from a mnemonic, seed, or private key.
*   **`StorageKey`:** Represents a storage key used to query chain state.  Can be created from a pallet, storage function, and parameters.
*   **`GenericCall`:** Represents a call to a runtime function.  Used to construct extrinsics.
*   **`GenericExtrinsic`:** Represents a signed transaction ready to be submitted to the chain.
*   **`ExtrinsicReceipt`:** Contains information about a submitted extrinsic, including its hash, block inclusion status, and triggered events.

### Connecting to a Node

```python
from commune import Chain

# Connect to a Substrate node using a URL
chain = Chain(url='wss://kusama-rpc.polkadot.io')

# Connect to a Substrate node using a network name
chain = Chain(network='kusama')
```

### Managing Keys

```python
from commune import Keypair

# Generate a keypair from a mnemonic
keypair = Keypair.create_from_mnemonic('bottom drive obey lake curtain smoke basket hold race lonely fit walk')

# Generate a keypair from a seed
keypair = Keypair.create_from_seed('0x0101010101010101010101010101010101010101010101010101010101010101')

# Generate a keypair from a private key
keypair = Keypair.create_from_private_key('0x0101010101010101010101010101010101010101010101010101010101010101')
```

### Querying Storage

```python
from commune import Chain

chain = Chain(network='main')

# Query a simple storage item
total_issuance = chain.query('TotalIssuance', module='Balances')
print(f"Total issuance: {total_issuance}")

# Query a mapped storage item
account_info = chain.query('Account', ['5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY'], module='System')
print(f"Account info: {account_info}")
```

### Submitting Extrinsics

```python
from commune import Chain, Keypair

chain = Chain(network='main')
keypair = Keypair.create_from_uri('//Alice')

# Compose a call
params = {'dest': '5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY', 'value': 1000}
receipt = chain.compose_call(fn='transfer_keep_alive', params=params, key=keypair, module='Balances')

if receipt['success']:
    print(f"Extrinsic submitted successfully! Extrinsic hash: {receipt['tx_hash']}")
else:
    print(f"Extrinsic failed: {receipt['error']}")
```

### Working with Multisig Accounts

```python
from commune import Chain, Keypair

chain = Chain(network='main')
keypair = Keypair.create_from_uri('//Alice')
signatories = [
    '5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY',  # Bob
    '5FHneW46xTkTDzSqDCbq9Vp6mGH6TdGBgqqBYtr9EBjeY8k',  # Charlie
    '5FLSigC9HGRKVhB9FiEo4Y3koPsNjndj935YQAidUhrFidj'   # Dave
]
threshold = 2

# Generate a multisig account
multisig_account = chain.generate_multisig_account(signatories, threshold)
print(f"Multisig account: {multisig_account.ss58_address}")

# Compose a call
params = {'dest': '5DAAnrj7VHTzbaCtTJ9alVvL6UG3PTAieyXoF6n4mQtH5eqg', 'value': 1000} # Eve's address
call = chain.compose_call(fn='transfer_keep_alive', params=params, module='Balances')

# Create a multisig extrinsic
receipt = chain.compose_call_multisig(
    fn='transfer_keep_alive',
    params=params,
    key=keypair,
    signatories=signatories,
    threshold=threshold,
    module='Balances'
)

if receipt['success']:
    print(f"Multisig extrinsic submitted successfully! Extrinsic hash: {receipt['tx_hash']}")
else:
    print(f"Multisig extrinsic failed: {receipt['error']}")
```

## API Reference

### `Chain` Class

*   **`__init__(network=network, url: str = None, mode = 'wss', num_connections: int = 1, wait_for_finalization: bool = False, test = False, ws_options = {}, timeout: int  = None, net = None)`:** Initializes a new `Chain` instance.
    *   `network`:  The name of the network to connect to (e.g., 'main', 'test', 'kusama').
    *   `url`:  The URL of the Substrate node.  If not provided, the module will use a default URL based on the `network`.
    *   `mode`: The connection mode ('wss' or 'https').
    *   `num_connections`: The number of connections to maintain in the connection pool.
    *   `wait_for_finalization`: Whether to wait for extrinsic finalization by default.
    *   `test`:  A boolean to use the test network.
    *   `ws_options`: A dictionary of options to pass to the websocket-client create\_connection function.
    *   `timeout`: Timeout for the websocket connection.
    *   `net`: Alias for network.
*   **`set_network(network=None, mode = 'wss', url = None, test = False, num_connections: int = 1, ws_options: dict = {}, wait_for_finalization: bool = False, timeout: int  = None)`:** Sets the network parameters.
*   **`get_conn(timeout: float = None, init: bool = False)`:** Context manager to get a connection from the connection pool.
*   **`get_storage_keys(storage: str, queries: list[tuple[str, list[Any]]], block_hash: str)`:** Gets the storage keys for a given storage and queries.
*   **`get_lists(storage_module: str, queries: list[tuple[str, list[Any]]], substrate: SubstrateInterface)`:** Generates a list of tuples containing parameters for each storage function.
*   **`rpc_request_batch(batch_requests: list[tuple[str, list[Any]]], extract_result: bool = True)`:** Sends batch requests to the Substrate node and collects the results.
*   **`rpc_request_batch_chunked(chunk_requests: list[Chunk], extract_result: bool = True)`:** Sends chunked batch requests to the Substrate node and collects the results.
*   **`query_batch(functions: dict[str, list[tuple[str, list[Any]]]]], block_hash: str = None, verbose=False)`:** Executes batch queries on a substrate and returns results in a 
dictionary format.
*   **`query_batch_map(functions: dict[str, list[tuple[str, list[Any]]]]], block_hash: str = None, path = None, max_age=None, update=False, verbose = False)`:** Queries multiple storage functions using a 
map batch approach and returns the combined result.
*   **`block_hash()`:** Returns the block hash.
*   **`block()`:** Returns the block number.
*   **`runtime_spec_version()`:** Returns the runtime spec version.
*   **`query(name: str, params: list[Any] = [], module: str = "SubspaceModule")`:** Queries a storage function on the network.
*   **`query_map(name: str='Emission', params: list[Any] = [], module: str = "SubspaceModule", extract_value: bool = True, max_age=60, update=False, block = None, block_hash: str = None)`:** Queries a storage map from a network node.
*   **`compose_call(fn: str, params: dict, key: Keypair, module: str = "SubspaceModule", wait_for_inclusion: bool = True, wait_for_finalization: bool = False, sudo: bool = False, tip = 0, nonce=None, unsigned: bool = False)`:** Composes and submits a call to the network node.
*   **`compose_call_multisig(fn: str, params: dict, key: Keypair, signatories: list[Ss58Address], threshold: int, module: str = "SubspaceModule", wait_for_inclusion: bool = True, wait_for_finalization: bool = None, sudo: bool = 
False, era: dict[str, int] = None)`:** Composes and submits a multisignature call to the network node.
*   **`transfer(key: Keypair, dest: Ss58Address, amount: int)`:** Transfers a specified amount of tokens from the signer's account to the specified account.
*   **`transfer_multiple(key: Keypair, destinations: list[Ss58Address], amounts: list)`:** Transfers multiple tokens to multiple addresses at once.
*   **`stake(key: Keypair, dest: Ss58Address, amount: int)`:** Stakes the specified amount of tokens to a module key address.
*   **`unstake(key: Keypair, dest: Ss58Address , amount: int)`:** Unstakes the specified amount of tokens from a module key address.
*   **`update_module(key: str, name: str=None, url: str = None, metadata: str = None, delegation_fee: int = None, validator_weight_fee = None, subnet = 2, public = False)`:** Updates a module.
*   **`register(name: str, url: str = '0.0.0.0:8000', module_key : str = None , key: Keypair = None, metadata: str = 'NA', subnet: str = 2, wait_for_finalization = False, public = False)`:** Registers a new module in the network.
*   **`deregister(key: Keypair, subnet: int=0)`:** Deregisters a module from the network.
*   **`register_subnet(name: str, metadata: str = None,  key: Keypair=None)`:** Registers a new subnet in the network.
*   **`vote(modules: list, weights: list, key: Keypair, subnet = 0)`:** Casts votes on a list of module UIDs with corresponding weights.
*   **`update_subnet(subnet, params: SubnetParams = None, **extra_params)`:** Update a subnet's configuration.
*   **`metadata(subnet=2)`:** Retrieves metadata.
*   **`stake_transfer(key: Keypair, from_module_key: Ss58Address, dest_module_address: Ss58Address, amount: int)`:** Realocate staked tokens from one staked module to another module.
*   **`multiunstake(key: Keypair, keys: list[Ss58Address], amounts: list)`:** Unstakes tokens from multiple module keys.
*   **`multistake(key: Keypair, keys: list[Ss58Address], amounts: list)`:** Stakes tokens to multiple module keys.
*   **`add_profit_shares(key: Keypair, keys: list[Ss58Address], shares: list)`:** Allocates profit shares to multiple keys.
*   **`add_subnet_proposal(key: Keypair, params: dict, ipfs: str, subnet: int = 0)`:** Submits a proposal for creating or modifying a subnet within the network.
*   **`add_custom_proposal(key: Keypair, cid: str)`:** Adds a custom proposal.
*   **`add_custom_subnet_proposal(key: Keypair, cid: str, subnet: int = 0)`:** Submits a proposal for creating or modifying a custom subnet within the network.
*   **`add_global_proposal(key: Keypair, params: NetworkParams, cid: str)`:** Submits a proposal for altering the global network parameters.
*   **`vote_on_proposal(key: Keypair, proposal_id: int, agree: bool)`:** Casts a vote on a specified proposal within the network.
*   **`unvote_on_proposal(key: Keypair, proposal_id: int)`:** Retracts a previously cast vote on a specified proposal.
*   **`enable_vote_power_delegation(key: Keypair)`:** Enables vote power delegation for the signer's account.
*   **`disable_vote_power_delegation(key: Keypair)`:** Disables vote power delegation for the signer's account.
*   **`add_dao_application(key: Keypair, application_key: Ss58Address, data: str)`:** Submits a new application to the general subnet DAO.
*   **`curator_applications()`:** Returns curator applications.
*   **`weights(subnet: int = 0, extract_value: bool = False )`:** Returns weights.
*   **`addresses( subnet: int = 0, extract_value: bool = False, max_age: int = 60, update: bool = False )`:** Returns addresses.
*   **`stake_from(key=None, extract_value: bool = False, fmt='j', **kwargs)`:** Retrieves a mapping of stakes from various sources for keys on the network.
*   **`stake_to( key=None, extract_value: bool = False, fmt='j', **kwargs )`:** Retrieves a mapping of stakes to destinations for keys on the network.
*   **`max_allowed_weights( extract_value: bool = False )`:** Retrieves a mapping of maximum allowed weights for the network.
*   **`legit_whitelist( extract_value: bool = False )`:** Retrieves a mapping of whitelisted addresses for the network.
*   **`subnet_names(extract_value: bool = False, max_age=60, update=False, block=None)`:** Retrieves a mapping of subnet names within the network.
*   **`proposal(proposal_id: int = 0)`:** Queries the network for a specific proposal.
*   **`proposals( extract_value: bool = False )`:** Returns proposals.
*   **`dao_treasury_address()`:** Returns the DAO treasury address.
*   **`global_dao_treasury()`:** Returns the global DAO treasury.
*   **`n( subnet: int = 0, max_age=60, update=False )`:** Queries the network for the 'N' hyperparameter.
*   **`total_stake(block_hash: str = None)`:** Retrieves a mapping of total stakes for keys on the network.
*   **`registrations_per_block()`:** Queries the network for the number of registrations per block.
*   **`unit_emission()`:** Queries the network for the unit emission setting.
*   **`tx_rate_limit()`:** Queries the network for the transaction rate limit.
*   **`subnet_burn()`:** Queries the network for the subnet burn value.
*   **`vote_mode_global()`:** Queries the network for the global vote mode setting.
*   **`max_proposals()`:** Queries the network for the maximum number of proposals allowed.
*   **`get_stakefrom(key: Ss58Address, fmt = 'j')`:** Retrieves the stake amounts from all stakers to a specific staked address.
*   **`get_staketo(key: Ss58Address = None, fmt = 'j')`:** Retrieves the stake amounts provided by a specific staker to all staked addresses.
*   **`balance(addr: Ss58Address=None, fmt = 'j')`:** Retrieves the balance of a specific key.
*   **`block()`:** Retrieves information about a specific block in the network.
*   **`existential_deposit(block_hash: str = None)`:** Retrieves the existential deposit value for the network.
*   **`voting_power_delegators()`:** Returns voting power delegators.
*   **`add_transfer_dao_treasury_proposal(key: Keypair, data: str, amount_nano: int, dest: Ss58Address)`:** Adds a transfer DAO treasury proposal.
*   **`delegate_rootnet_control(key: Keypair, dest: Ss58Address)`:** Delegates rootnet control.
*   **`to_nano(value)`:** Converts a value to nanotokens.
*   **`to_joules(value)`:** Converts a value to joules.
*   **`valid_h160_address(address)`:** Validates if the address is a valid H160 address.
*   **`resolve_key_address(key:str )`:** Resolves the key address.
*   **`resolve_key(key:str )`:** Resolves the key.
*   **`params(subnet = None, block_hash: str = None, max_age=tempo,  update=False)`:** Gets all subnets info on the network.
*   **`global_params(max_age=60, update=False)`:** Returns global parameters of the whole commune ecosystem.
*   **`founders()`:** Returns founders.
*   **`my_subnets(update=False)`:** Returns my subnets.
*   **`my_modules(subnet="all", max_age=60, keys=None, features=['name', 'key', 'url', 'emission', 'weights', 'stake'], df = False, update=False)`:** Returns my modules.
*   **`my_valis(subnet=0, min_stake=0)`:** Returns my validators.
*   **`my_keys(subnet=0)`:** Returns my keys.
*   **`valis(subnet=0, max_age=600, update=False, df=1, search=None, min_stake=0, features=['name', 'key', 'stake_from', 'weights'], **kwargs)`:** Returns validators.
*   **`storage2name(name)`:** Converts storage to name.
*   **`name2storage(name, name_map={'url': 'address'})`:** Converts name to storage.
*   **`modules(subnet=2, max_age = tempo, update=False, timeout=30, module = "SubspaceModule", features = ['key', 'url', 'name', 'metadata'], lite = True, num_connections = 1, search=None, df = False, 
**kwargs)`:** Returns modules.
*   **`format_amount(self, x, fmt='nano')`:** Formats an amount.
*   **`netuids(self,  update=False, block=None)`:** Returns netuids.
*   **`emissions(self, **kwargs )`:** Returns emissions.
*   **`subnet2netuid(self, **kwargs )`:** Returns subnet to netuid.
*   **`netuid2subnet(self, update=False, block=None, max_age=None)`:** Returns netuid to subnet.
*   **`is_registered(self, key=None, subnet=0,max_age=60)`:** Returns if a key is registered.
*   **`module(self, module, subnet=2,fmt='j', mode = 'https', block = None, **kwargs )`:** Returns a module.
*   **`vec82str(x)`:** Converts vec8 to string.
*   **`storage(search=None, pallets= pallets, features = ['name', 'modifier', 'type', 'docs'])`:** Returns storage.
*   **`test()`:** Runs tests.

### `Keypair` Class

*   **`__init__(ss58_address: str = None, public_key: Union = None, private_key: Union = None, ss58_format: int = None, seed_hex: Union = None, crypto_type: int = KeypairType.SR25519)`:** Initializes a new `Keypair` instance.
*   **`generate_mnemonic(words: int = 12, language_code: str = MnemonicLanguageCode.ENGLISH)`:** Generates a new seed phrase with given amount of words.
*   **`validate_mnemonic(mnemonic: str, language_code: str = MnemonicLanguageCode.ENGLISH)`:** Verify if specified mnemonic is valid.
*   **`create_from_mnemonic(mnemonic: str, ss58_format=42, crypto_type=KeypairType.SR25519, language_code: str = MnemonicLanguageCode.ENGLISH)`:** Create a Keypair for given memonic.
*   **`create_from_seed(seed_hex: Union, ss58_format: Optional = 42, crypto_type=KeypairType.SR25519)`:** Create a Keypair for given seed.
*   **`create_from_uri(suri: str, ss58_format: Optional = 42, crypto_type=KeypairType.SR25519, language_code: str = MnemonicLanguageCode.ENGLISH)`:** Creates Keypair for specified suri in following format: `///[hard-path]`.
*   **`create_from_private_key(private_key: Union, public_key: Union = None, ss58_address: str = None, ss58_format: int = None, crypto_type: int = KeypairType.SR25519)`:** Creates Keypair for specified public/private keys.
*   **`create_from_encrypted_json(json_data: Union, passphrase: str, ss58_format: int = None)`:** Create a Keypair from a PolkadotJS format encrypted JSON file.
*   **`export_to_encrypted_json(passphrase: str, name: str = None)`:** Export Keypair to PolkadotJS format encrypted JSON file.
*   **`sign(data: Union[ScaleBytes, bytes, str])`:** Creates a signature for given data.
*   **`verify(data: Union[ScaleBytes, bytes, str], signature: Union)`:** Verifies data with specified signature.
*   **`encrypt_message(message: Union, recipient_public_key: bytes, nonce: bytes = secrets.token_bytes(24),)`:** Encrypts message with for specified recipient.
*   **`decrypt_message(encrypted_message_with_nonce: bytes, sender_public_key: bytes)`:** Decrypts message from a specified sender.

## Types

The module defines several type aliases and `TypedDict`s to represent common data structures used in Substrate-based chains:

*   `Ss58Address`:  A string representing a Substrate SS58 address.
*   `NetworkParams`:  A `TypedDict` representing global network parameters.
*   `SubnetParams`:  A `TypedDict` representing subnet parameters.
*   `SubnetParamsWithEmission`:  A `TypedDict` representing subnet parameters with an emission field.
*   `ModuleInfo`: A `TypedDict` representing module information.
*   `ChainTransactionError`: An exception for chain transaction related errors.
*   `NetworkQueryError`: An exception for network query related errors.

## Exceptions

The module defines several custom exceptions to handle specific error conditions:

*   `SubstrateRequestException`:  Base exception for RPC request errors.
*   `StorageFunctionNotFound`:  Raised when a storage function is not found in the metadata.
*   `ConfigurationError`:  Raised for configuration-related errors.
*   `ExtrinsicFailedException`:  Raised when an extrinsic fails.
*   `BlockNotFound`:  Raised when a block is not found.
*   `ExtrinsicNotFound`:  Raised when an extrinsic is not found in a block.
*   `ExtensionCallNotFound`:  Raised when an extension call is not found.

## Contributing

Contributions are welcome! Please submit pull requests with clear descriptions of the changes and justifications for their inclusion.

## License

This module is licensed under the Apache License 2.0. See the `LICENSE` file for more information.