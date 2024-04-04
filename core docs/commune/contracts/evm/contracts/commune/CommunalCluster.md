# CommunalCluster Smart Contract

The "CommunalCluster" smart contract is an implementation of the ERC20 token standard with custom additions. It extends the functionality of a basic ERC20 token with communal resources that can be added and managed by the contract's owner.

## Features

### Basic ERC20 functionalities

- Transfer tokens: Users can transfer their tokens to other Ethereum addresses.
- Approve tokens for spending: Users can approve a certain amount of their tokens to be spent by others.
- Allowances: This contract keeps track of the allowance that an address has given to another address.
- Increase/decrease allowances: Allowances can be increased or decreased atomically in a single transaction, which can avoid some potential race condition problems.

### Extentions

- Add resource: The contract's owner can add resources, including their price rate and URI for accessing them.
- Deposit: Any user can deposit Ether to mint new tokens. The amount of tokens that will be minted is the deposited Ether multiplied by 10^18 (as the tokens have 18 decimals).
- Withdraw:  The owner of the contract has the ability to withdraw Ether from the contract.


## Usage

### Constructor

When deploying the smart contract, the `name` and `symbol` of the token have to be specified:

```javascript
constructor(string memory name_, string memory symbol_)
```

### Add Resource

The owner can add a resource by specifying its name, price rate and URI:

```javascript
function add_resource(string memory name, uint256 price_rate, string memory uri) public
```

### Deposit

Users can deposit Ether and will receive tokens in return according to the contract's conversion rate:

```javascript
function deposit() public payable
```

### Withdraw

The owner can withdraw Ether from the contract. The `amount` specified is in wei:

```javascript
function withdraw(uint256 amount) public payable
```

### Basic ERC20 functions

- `transfer(address to, uint256 amount)`
- `approve(address spender, uint256 amount)`
- `increaseAllowance(address spender, uint256 addedValue)`
- `decreaseAllowance(address spender, uint256 subtractedValue)`

## Disclaimer

Please note that this contract was last updated for OpenZeppelin Contracts v4.7.0 and Solidity 0.8, and may not be compatible with more recent versions. Always review and test smart contracts thoroughly before using them.
