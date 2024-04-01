# Address library in Solidity by OpenZeppelin

This library provides a set of utility functions related to the Ethereum address datatype.

## Functionality

1. `isContract(address account)`: This function can be used to detect if an Ethereum address is a contract.

2. `sendValue(address payable recipient, uint256 amount)`: Sends `amount` of ether to `recipient`. This function forwards all available gas and reverts on errors.

3. `functionCall(address target, bytes memory data)`: Performs a low level `call` to `target` with `data`.

4. `functionCallWithValue(address target, bytes memory data, uint256 value)`: Similar to `functionCall`, but also transfers `value` of ether to `target`.

5. `functionStaticCall(address target, bytes memory data)`: Performs a low level static `call` to `target` with `data`.

6. `functionDelegateCall(address target, bytes memory data)`: Performs a low level `delegatecall` to `target` with `data`.

7. Several variants of the above function calls exist that accept a `errorMessage` parameter, which will be used as a fallback revert reason when `target` reverts.

## Important Notes

It is critical to use these functions safely; the comments in the code and official documentation should be taken into account for a secure integration. In particular, care must be taken to not create reentrancy vulnerabilities, and it is recommended to use the [checks-effects-interactions pattern](https://solidity.readthedocs.io/en/v0.5.11/security-considerations.html#use-the-checks-effects-interactions-pattern).

## Dependencies

- Solidity ^0.8.1
- OpenZeppelin ^4.7.0

## Documentation
- [OpenZeppelin Documentation](https://docs.openzeppelin.com/contracts/4.x/api/utils#Address)