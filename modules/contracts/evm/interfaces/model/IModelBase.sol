// SPDX-License-Identifier: MIT OR Apache-2.0
pragma solidity ^0.8.7;

contract IModelNFT {
    // function getState() external virtual view returns(address, address, string memory, string memory, uint256, string memory, uint256){}
    function processPayment(address userAddress) external virtual payable {}
    
}
