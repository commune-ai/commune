// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts v4.4.1 (access/AccessControl.sol)

pragma solidity ^0.8.0;

import "./IAccessControl.sol";
import "contracts/utils/context/ContextAdapter.sol";

interface  IAccessControlAdapter{
    
    function connectAccessController(address ac_acccount) external;
    function hasRole(string memory role, address account) external returns(bool);
    
}
