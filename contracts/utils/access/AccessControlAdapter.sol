// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts v4.4.1 (access/AccessControl.sol)

pragma solidity ^0.8.0;

import "interfaces/utils/access/IAccessControl.sol";
import "interfaces/utils/access/IAccessControlAdapter.sol";
import "contracts/utils/context/ContextAdapter.sol";

contract  AccessControlAdapter is IAccessControlAdapter{


    IAccessControl public accessController ;
    
    modifier onlyRole(string memory role,address account) {
        require(accessController.hasRole(role, account));
        _;
    }

    modifier onlyAdmin(address account) {
        require(hasRole("owner", account), "the account is not the admin");
        _;
    }
    
    function connectAccessController(address ac_acccount) public override  {
        accessController =  IAccessControl(ac_acccount);
    }
    function hasRole(string memory role, address account) public view override returns(bool){
        return accessController.hasRole(role, account);
    }



}
