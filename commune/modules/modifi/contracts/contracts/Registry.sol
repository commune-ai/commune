// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract Registry {
    event Added(address indexed market, address indexed creator);
    address[] public markets; mapping(address=>bool) public isRegistered;
    function add(address market, address creator) external { require(!isRegistered[market],"exists"); isRegistered[market]=true; markets.push(market); emit Added(market,creator);}    
    function count() external view returns(uint256){ return markets.length; }
    function getAll() external view returns(address[] memory){ return markets; }
}
