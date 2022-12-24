// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts v4.4.1 (access/AccessControl.sol)

pragma solidity ^0.8.0;

import "interfaces/utils/access/IAccessControl.sol";
import "contracts/utils/context/Context.sol";
import "@openzeppelin/contracts/utils/Strings.sol";
import "contracts/utils/context/ContextAdapter.sol";


contract AccessControl is  IAccessControl, Context {
    struct RoleData {
        mapping(address => bool) members;
        uint256 limit;
        uint256 count;
    }

    modifier onlyAdmin() {
        _checkRole(ADMIN_ROLE, _msgSender());
        _;
    }

    modifier onlyRole(string memory role) {
        _checkRole(role, _msgSender());
        _;
    }

    mapping(string => RoleData) private _roles;
    string public ADMIN_ROLE ="owner";
    function getAdminRole() public view override returns(string memory){
        string memory _admin_role  = ADMIN_ROLE;
        return _admin_role;
    }

    constructor() {
        _grantRole(ADMIN_ROLE, _msgSender());
    }


    function _grantRole(string memory role, address account) internal virtual {

        if (!hasRole(role, account)) {
            _roles[role].members[account] = true;
            _roles[role].count++;
            // emit RoleGranted(role, account, _msgSender());
        }
    }

    function _revokeRole(string memory role, address account) internal virtual {
        if (hasRole(role, account)) {
            _roles[role].members[account] = false;
            _roles[role].count--;
            // emit RoleRevoked(role, account, _msgSender());
        }
    }

    function _checkRole(string memory role, address account) internal view {
        if (!hasRole(role, account)) {
            revert(
                string(
                    abi.encodePacked(
                        "AccessControl: account ",
                        Strings.toHexString(uint160(account), 20),
                        " is missing role ", 
                        role
                    )
                )
            );
        }
    }


    function hasRole(string memory role, address account) public view override returns (bool) {
        return _roles[role].members[account];
    }

    function grantRole(string memory role, address account) public virtual override onlyAdmin {
        _grantRole(role, account);
    }

    function revokeRole(string memory role, address account) public virtual override onlyAdmin {
        _revokeRole(role, account);
    }

    function renounceRole(string memory role, address account) public virtual override {
        require(account == _msgSender(), "AccessControl: can only renounce roles for self");
        _revokeRole(role, account);
    }
}
