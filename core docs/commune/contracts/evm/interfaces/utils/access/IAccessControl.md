# OpenZeppelin's IAccessControl Interface

This is a solidity interface for OpenZeppelin's Access Control contract. The Access Control contract is a flexible, role-based access control mechanism that checks whether an account has been granted a role. Each account can be assigned multiple roles, and roles can have multiple members.

## Features:

- **RoleAdminChanged Event:** Emitted when a new role is set as the admin role replacing the `previousAdminRole`. 'DEFAULT_ADMIN_ROLE' is the starting admin for all roles.

- **RoleGranted Event**: Emitted when an account is granted a role. The `sender` account is the one that grants the role.

- **RoleRevoked Event**: Emitted when a role is revoked from an account. `sender` is the account that revokes the role.

- **hasRole Function**: Checks If a role has been granted to an account.

- **getAdminRole Function**: Returns the admin role for a provided role.

- **grantRole Function**: Grants a role to an account.

- **revokeRole Function**: Revokes a role from an account.

- **renounceRole Function**: An account can renounce a role to lose its privileges.

## Prerequisite
To use `IAccessControl` interface, import it in your smart contract.

## Example:
```javascript
pragma solidity ^0.8.0;
import "@openzeppelin/contracts/access/IAccessControl.sol";

contract MyContract implements IAccessControl {
    //...
    function doSomething() public {
        require(hasRole(DEFAULT_ADMIN_ROLE, msg.sender), "Caller is not an admin");
        //...
    }
    //...
}
```
