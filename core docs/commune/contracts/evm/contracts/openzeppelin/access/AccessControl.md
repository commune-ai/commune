# AccessControl.sol

AccessControl is a module for role-based authorization control. It allows to implement roles and grant permissions for those roles to Ethereum public addresses. It is very useful for controlling permissions for different sections of a smart contract.

## Important Definitions

- Role: It is a set of permissions. It is referred to by its identifier, which is represented in `bytes32`.
- Account: An Ethereum public address.
- Admin Role: For every role in the contract, there is an associated admin role.

Role-based access control is flexible and easy to use. The role identifier must be unique and can be obtained by hashing a relevant role descriptor text.

## Role-based access control

You can check if an account has a particular role as below:

``` solidity
  function foo() public {
   require(hasRole(MY_ROLE, msg.sender));
   ...
  }
```

The example above checks for `MY_ROLE` in the calling account (msg.sender).

### Granting a role

To grant a role to a certain account, call the function:

```solidity
    function grantRole(bytes32 role, address account) public virtual override onlyRole(getRoleAdmin(role)) {
        _grantRole(role, account);
    }
```
A role can only be granted by an address that has the admin role for the role being granted.

### Revoking a role

To remove a role from a certain account:

```solidity
    function revokeRole(bytes32 role, address account) public virtual override onlyRole(getRoleAdmin(role)) {
        _revokeRole(role, account);
    }
```
A role can be revoked only by an address that holds the admin role for the role in question.

### Renouncing a role

An address can voluntarily relinquish a role it has:

```solidity
  function renounceRole(bytes32 role, address account) public virtual override {
    require(account == _msgSender(), "AccessControl: can only renounce roles for self");

    _revokeRole(role, account);
  }
```

This function is mainly for scenarios where a privileged account is possibly compromised.

## Admin Roles

`DEFAULT_ADMIN_ROLE` is the admin role that is able to grant and revoke any other role. Be cautious who is granted this role.

You can get admin role of any role by calling `getRoleAdmin()` function.

You can set a specific admin role for a specific role using `_setRoleAdmin()` (modifier `internal`).

## Events

- RoleGranted: Emitted when a role is granted to an account.
- RoleRevoked: Emitted when a role is revoked from an account.
- RoleAdminChanged: Emitted when the admin role of a role is changed.
