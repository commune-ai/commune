# Minimal Forwarder Smart Contract

This smart contract is a minimalist forwarder principally intended for testing together with an ERC2771 compatible contract and takes advantage of EIP712 typology. 

Please note: due to its simplicity, this contract is **not** suitable as a robust, production-ready forwarder, and for a more complex, hands-on tool, you should refer to programs such as the Gas Station Network (GSN).

## Features

- **Get Nonce:** You can retrieve a nonce for a specific account.
- **Verify:** This function verifies if the provided signature matches with the signed data hashed according to EIP712.
- **Execute:** This function executes a call to another contract if the provided signature is valid and matches the request.

The forward request struct includes the following variables:
- The address of the sender and receiver
- Value to be sent
- Gas for the execution
- Nonce to prevent replay attacks 
- Data to be sent in the execution

## Interfaces and Inheritance

The contract extends from `EIP712`.

- **EIP712:** A Solidity implementation providing the ability to verify signatures according to the EIP-712 standard.
