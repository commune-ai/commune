// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts v4.4.1 (utils/Context.sol)
pragma solidity ^0.8.0;

import "contracts/utils/context/Context.sol";

abstract contract ContextAdapter {
    Context public context;
    function connectContext(address contextAddress) public {
        context =  Context(contextAddress);
    }
}
